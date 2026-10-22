"""Sharded DFlash projections and the checkpoint's learned mask embedding.

With the "w8a16" weight format every projection becomes block-FP8 (E4M3 values, one FP32
scale of amax / 448 per 128x128 block) after slicing to this rank's shard; each shard
boundary is 128-aligned, so the blocks are those of the full matrix.  Activations stay BF16.
"""

from dataclasses import dataclass, replace

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint

# "w8a16": block-FP8 projections run by W8A16 GEMVs; "bf16": BF16 projections run by b12x.
WEIGHT_FORMATS = ("w8a16", "bf16")


@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: object
    output: object
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    sink: torch.Tensor | None
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class DraftWeights:
    projection: object
    hidden_norm: torch.Tensor
    context_kv: object
    norm: torch.Tensor
    mask_embedding: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]

    @classmethod
    def load(cls, snapshot, config, device, shard, *, progress=None, weight_format="bf16"):
        if weight_format not in WEIGHT_FORMATS:
            raise ValueError(f"DFlash weight format must be one of {WEIGHT_FORMATS}")
        c, a = config.text, config.attention
        q_range = shard.partition(a.heads * a.key_dim, alignment=a.key_dim)
        first, last = shard.replicated_partition(a.kv_heads)
        k_range = first * a.key_dim, last * a.key_dim
        v_range = first * a.value_dim, last * a.value_dim
        intermediate = shard.partition(c["intermediate_size"], alignment=128)
        manifest = inspect_indexed_checkpoint(snapshot)
        layers, context = [], []
        with DirectWeightLoader(manifest, device, progress=progress, progress_label="Load draft weights") as loader:
            def matrix(name, rows=None, columns=None):
                value = loader.load_tensor(name + ".weight", extent=(slice(*(rows or (None, None))),
                                                                     slice(*(columns or (None, None)))))
                if value.dtype != torch.bfloat16 or value.ndim != 2:
                    raise ValueError(f"DFlash requires BF16 matrices: {name}")
                return value

            projection = matrix("fc", rows=shard.partition(c["hidden_size"], alignment=64))
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            for i in range(config.layers):
                prefix = f"layers.{i}"
                qkv = torch.cat([matrix(prefix + ".self_attn." + name, rows=rows)
                                 for name, rows in (("q_proj", q_range), ("k_proj", k_range), ("v_proj", v_range))])
                q_rows = q_range[1] - q_range[0]
                context.append(qkv[q_rows:])
                output = matrix(prefix + ".self_attn.o_proj",
                                columns=shard.partition(a.heads * a.value_dim, alignment=a.value_dim))
                query_norm, key_norm, input_norm, post_norm = loader.load_tensors(tuple(prefix + name for name in
                    (".self_attn.q_norm.weight", ".self_attn.k_norm.weight", ".input_layernorm.weight",
                     ".post_attention_layernorm.weight")))
                sink = loader.load_tensor(prefix + ".self_attn.attention_sink_bias", dtype=torch.float32,
                    extent=slice(*shard.partition(a.heads))) if a.sinks else None
                mlp = MLPWeights(torch.cat([matrix(prefix + ".mlp." + name, rows=intermediate)
                                           for name in ("gate_proj", "up_proj")]),
                                 matrix(prefix + ".mlp.down_proj", columns=intermediate))
                layers.append(DraftLayerWeights(qkv, output, query_norm, key_norm, sink, input_norm, post_norm, mlp))
        mask = torch.load(snapshot / "mask_embedding.pt", map_location="cpu", weights_only=True)
        if (not isinstance(mask, dict) or mask.get("mask_token_id") != config.noise_token
                or not isinstance(mask.get("embedding"), torch.Tensor)
                or mask["embedding"].shape != (c["hidden_size"],)
                or mask["embedding"].dtype != torch.bfloat16):
            raise ValueError("DFlash mask_embedding.pt disagrees with the draft configuration")
        weights = cls(projection, hidden_norm, torch.cat(context), norm, mask["embedding"].to(device), tuple(layers))
        return weights if weight_format == "bf16" else weights.block_fp8()

    @staticmethod
    def serving_format(weight_format, config, requests):
        """The weight format for drafts of ``requests`` blocks: W8A16 GEMVs serve at most 32 rows
        per launch, so larger draft batches, which would re-read the weights per 32-row slice,
        keep BF16; their smaller graphs then use a block-FP8 copy (ModelWeights.draft_w8a16)."""
        from ignition.components.w8a16_gemv import MAX_ROWS

        if weight_format == "w8a16" and requests is not None and requests * config.block_size > MAX_ROWS:
            return "bf16"
        return weight_format

    def block_fp8(self):
        """These weights with every projection quantized to block-FP8 for W8A16 GEMVs."""
        from ignition.components.w8a16_gemv import quantize_block_fp8

        layers = tuple(replace(layer, qkv=quantize_block_fp8(layer.qkv), output=quantize_block_fp8(layer.output),
            mlp=MLPWeights(quantize_block_fp8(layer.mlp.gate_up), quantize_block_fp8(layer.mlp.down)))
            for layer in self.layers)
        return replace(self, projection=quantize_block_fp8(self.projection),
                       context_kv=quantize_block_fp8(self.context_kv), layers=layers)

    @property
    def projection_shapes(self):
        """(N, K) of every projection, for sizing shared W8A16 scratch."""
        matrices = [self.projection, self.context_kv]
        for layer in self.layers:
            matrices += [layer.qkv, layer.output, layer.mlp.gate_up, layer.mlp.down]
        return tuple(sorted({tuple(_values(m).shape) for m in matrices}))


def _values(weight):
    # A BF16 tensor's ``values`` is the sparse-tensor method, not a weight plane.
    return weight if isinstance(weight, torch.Tensor) else weight.values


def projection_rows(weight):
    """Output features of a BF16 or block-FP8 projection."""
    return _values(weight).shape[0]
