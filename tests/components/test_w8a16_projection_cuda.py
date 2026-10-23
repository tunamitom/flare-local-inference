"""W8A16 projections over block-FP8 weights: compilation, shared split-K scratch and graph replay."""

import os

import pytest
import torch

from ignition.components.w8a16_gemv import W8A16Projection, W8A16Workspace, quantize_block_fp8
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


def _dequantized(weight):
    n, k = weight.values.shape
    scales = weight.scales.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return weight.values.float() * scales


@pytest.mark.parametrize("rows", [1, 8, 32, 70])
def test_w8a16_projections_share_scratch_through_compiled_replay(rows):
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(rows)
    # The drafter's TP8 qkv and down shapes; one row split crosses the 32-row launch limit.
    first = quantize_block_fp8(torch.randn((2304, 6144), device=device, generator=generator).to(torch.bfloat16))
    second = quantize_block_fp8(torch.randn((6144, 2048), device=device, generator=generator).to(torch.bfloat16))
    source = torch.randn((rows, 6144), device=device, dtype=torch.bfloat16, generator=generator)
    middle = torch.empty((rows, 2304), device=device, dtype=torch.bfloat16)
    narrowed = torch.empty((rows, 2048), device=device, dtype=torch.bfloat16)
    output = torch.empty((rows, 6144), device=device, dtype=torch.bfloat16)
    workspace = W8A16Workspace(((2304, 6144), (6144, 2048)), device)
    # Without a recorded session the launches prepare at construction.
    up = W8A16Projection(source, first, middle, session=None, name="up", workspace=workspace)
    down = W8A16Projection(narrowed, second, output, session=None, name="down", workspace=workspace)

    def forward():
        narrowed.copy_(up(source)[:, :2048])
        down(narrowed)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for multiplier in (1., .25, 3.):
        source.mul_(multiplier)
        hidden = (source.float() @ _dequantized(first).T).to(torch.bfloat16)
        expected = hidden[:, :2048].float() @ _dequantized(second).T
        graph.replay()
        torch.cuda.synchronize()
        reference = output.clone()
        torch.testing.assert_close(output.float(), expected, rtol=.02, atol=.02 * expected.abs().max().item())
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, reference)
        assert not workspace.counters.any()
    graph.reset()


@pytest.mark.parametrize("rows", [1, 8, 16, 32])
def test_draft_head_shard_uses_its_tuned_configs_through_graph_replay(rows):
    from ignition.components.w8a16_gemv import default_config

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(rows)
    # MiMo's TP8 vocabulary shard: 152,576 tokens over eight ranks.
    head = quantize_block_fp8((torch.randn((19072, 6144), device=device, generator=generator) * .02)
                              .to(torch.bfloat16))
    assert default_config(rows, 19072, 6144).block_n == (128 if rows <= 8 else 64)
    source = torch.randn((rows, 6144), device=device, dtype=torch.bfloat16, generator=generator)
    logits = torch.empty((rows, 19072), device=device, dtype=torch.bfloat16)
    workspace = W8A16Workspace(((19072, 6144),), device)
    project = W8A16Projection(source, head, logits, session=None, name="head", workspace=workspace)
    graph = torch.cuda.CUDAGraph()
    project(source)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        project(source)
    graph.replay()
    torch.cuda.synchronize()
    expected = source.float() @ _dequantized(head).T
    torch.testing.assert_close(logits.float(), expected, rtol=.02, atol=.02 * expected.abs().max().item())
    assert not workspace.counters.any()
    graph.reset()


def test_quantization_layout_matches_amax_scales():
    device = torch.device("cuda", 0)
    weight = torch.randn((200, 256), device=device).to(torch.bfloat16)
    quantized = quantize_block_fp8(weight)
    assert quantized.values.shape == (200, 256) and quantized.scales.shape == (2, 2)
    expected = weight[:128, :128].float().abs().amax() / 448
    torch.testing.assert_close(quantized.scales[0, 0], expected)
    assert quantized.values[:128, :128].float().abs().amax() == 448


class _Stack:
    def __init__(self, source, weights, device):
        self.source = source
        self.middle = torch.empty((source.shape[0], 768), device=device, dtype=torch.bfloat16)
        self.output = torch.empty((source.shape[0], 1280), device=device, dtype=torch.bfloat16)
        workspace = W8A16Workspace(((768, 30720), (1280, 768)), device)
        self.project = W8A16Projection(source, weights[0], self.middle, session=None, name="project",
                                       workspace=workspace)
        self.project_kv = W8A16Projection(self.middle, weights[1], self.output, session=None, name="kv",
                                          workspace=workspace)

    def __call__(self):
        self.project_kv(self.project(self.source))


def test_w8a16_aot_storage_binding_and_serialization(tmp_path):
    from ignition.preparation.aot import aot_forward

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(5)
    weights = (quantize_block_fp8(torch.randn((768, 30720), device=device, generator=generator).to(torch.bfloat16)),
               quantize_block_fp8(torch.randn((1280, 768), device=device, generator=generator).to(torch.bfloat16)))
    stack = _Stack(torch.randn((40, 30720), device=device, dtype=torch.bfloat16, generator=generator), weights,
                   device)
    stack()
    expected = stack.output.clone()
    compiled = aot_forward(stack)
    stack.output.fill_(float("nan"))
    compiled(stack)
    assert torch.equal(stack.output, expected)
    saved = tmp_path / "w8a16.bin"
    compiled.save_compiled_function(str(saved))
    with saved.open("rb") as source:
        restored = torch.compiler.load_compiled_function(source)
    assert restored.guard_check(stack)
    stack.output.fill_(float("nan"))
    restored(stack)
    assert torch.equal(stack.output, expected)


def test_prepared_mlp_runs_block_fp8_weights_as_w8a16():
    from ignition.components.mlp import PreparedMLP
    from ignition.components.weights import MLPWeights
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(11)
    rows, hidden, width = 24, 1024, 512
    gate_up = torch.randn((2 * width, hidden), device=device, generator=generator).to(torch.bfloat16) / 32
    down = torch.randn((hidden, width), device=device, generator=generator).to(torch.bfloat16) / 16
    weights = MLPWeights(quantize_block_fp8(gate_up), quantize_block_fp8(down))
    source = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16, generator=generator)
    buffers = [torch.empty((rows, n), device=device, dtype=torch.bfloat16) for n in (2 * width, width, hidden)]
    workspace = W8A16Workspace(((2 * width, hidden), (hidden, width)), device)
    with RecordedPreparationSession(device=device, checkpoint="w8a16-mlp", workload_schema="w8a16-mlp",
                                    numerical_policy="default") as session:
        mlp = PreparedMLP(source, weights, *buffers, session=session, name="mlp", workspace=None, w8a16=workspace)
        assert all(isinstance(p, W8A16Projection) for p in (mlp.up_projection, mlp.down_projection))
        session.tune()
        output = mlp(source).float()
    projected = (source.float() @ _dequantized(weights.gate_up).T).to(torch.bfloat16).float()
    gate, up = projected.chunk(2, -1)
    intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
    expected = intermediate @ _dequantized(weights.down).T
    torch.testing.assert_close(output, expected, rtol=.02, atol=.02 * expected.abs().max().item())
