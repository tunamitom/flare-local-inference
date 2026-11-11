"""Preflight: Dynamo fullgraph smoke of the graph-touching W22 riders.

The export's aot_forward fullgraph-captures every model forward; a rider that
touches the traced path with an untraceable op (e.g. a data-pointer comparison)
fails the whole export AFTER the weights load and the kernel compiles. This
smoke compiles each rider's hot call under torch.compile(fullgraph=True) on the
spare card (tiny inputs) and fails fast if Dynamo rejects it.

New graph-touching riders MUST add a case here (same pattern: a compiled
wrapper, an eager reference, and the eager contract guards).

Run under flock /tmp/gpu-spare.lock with CUDA_VISIBLE_DEVICES set to the spare
card (GPU identity: switch port + PCI BDF 11:00.0). Leaves no state.
"""
import torch

from ignition.components.distributed_topk import selected_local_records


def case_t17_compaction():
    """W22 retained-buffer compaction rider (selected_local_records out=/count_out=)."""
    torch.manual_seed(0)
    rows, width, world, rank, page_size = 8, 16, 8, 3, 32
    selected = torch.randint(-4, 512, (rows, width)).to(torch.int32).cuda()
    positions = torch.full((rows,), 400, dtype=torch.int32).cuda()
    pages = torch.randint(0, 97, (rows, 16)).to(torch.int32).cuda()
    out_buf = torch.empty_like(selected)
    count_buf = torch.empty((rows,), dtype=torch.int32).cuda()

    def fwd(selected, positions, pages, out_buf, count_buf):
        return selected_local_records(selected, positions, pages, world=world, rank=rank,
                                      page_size=page_size, out=out_buf, count_out=count_buf)

    ref_out, ref_count = selected_local_records(selected, positions, pages,
                                                world=world, rank=rank, page_size=page_size)
    got_out, got_count = torch.compile(fwd, fullgraph=True)(
        selected, positions, pages, out_buf, count_buf)
    torch.cuda.synchronize()
    assert torch.equal(ref_out, got_out), "compiled compaction differs from eager (out)"
    assert torch.equal(ref_count, got_count), "compiled compaction differs from eager (count)"

    fired = 0
    for kwargs in (
        dict(out=selected, count_out=count_buf),
        dict(out=out_buf, count_out=out_buf.reshape(-1)[:rows]),
    ):
        try:
            selected_local_records(selected, positions, pages, world=world, rank=rank,
                                   page_size=page_size, **kwargs)
        except ValueError:
            fired += 1
    assert fired == 2, f"eager T17 guards fired {fired}/2"
    return "compiled == eager; T17 guards 2/2"


def case_w23_route_pack_flow():
    """W23 rider-(a) knob plumbing (in-kernel route pack + flow counters).

    The two knobs are read at fused-kernel construction (behind the b12x
    custom-op boundary), so the Dynamo risk is the op boundary itself: with the
    knob env set, the prepared chain at the W23 unmeasured 20-row shape must
    compile under fullgraph exactly like eager and stay bit-identical. Also
    checks the construction contract: the flow knob must extend the planned
    kernel workspace (w4a16 flow workspace), i.e. the knob demonstrably engages
    the planner. Construction/identity only — no timing. The W23-C export ships
    with both switches UNSET (rider-(a) failed the J-G 20-row speed bar), but
    the knob path stays smoke-covered for any future armed export.
    """
    import hashlib
    import os

    from b12x.moe import fused_moe
    from b12x.moe.checkpoints.exl3 import (
        read_exl3_layer,
        read_exl3_manifest,
        trellis_from_exl3,
    )
    from b12x.moe._shared.kernels.w4a16.kernel.types import W4A16_FLOW_WORKSPACE_EXTRA
    from b12x.preparation import PreparedCall, PreparationSession
    from b12x.preparation.types import require_prepared

    knobs = ("B12X_TRELLIS_INKERNEL_ROUTE_PACK", "B12X_TRELLIS_FLOW_FLAGS")
    saved = {name: os.environ.get(name) for name in knobs}

    def digest(t):
        return hashlib.sha256(
            t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    def build_and_run(engage):
        for name in knobs:
            os.environ.pop(name, None)
        if engage:
            os.environ["B12X_TRELLIS_INKERNEL_ROUTE_PACK"] = "2"
            os.environ["B12X_TRELLIS_FLOW_FLAGS"] = "1"
        session = PreparationSession(
            device=torch.device("cuda"), autotune=False, compile_workers=0
        )
        try:
            decl = fused_moe.plan_execution(
                experts=experts,
                capacity=fused_moe.ExecutionCapacity(max_tokens=M, top_k=K),
            )

            def primer(st):
                sc = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=s.device)
                    for s in st.scratch.scratch_specs()
                )
                b = st.bind(
                    a=x, experts=experts, topk_ids=ids, topk_weights=w,
                    output=out, scratch=sc, input_scales_static=True,
                )
                return PreparedCall(run=lambda: st.run(b), owners=sc)

            req = decl.request(name="smoke", prepare_call=primer)
            result = session.prepare((req,))
            st = require_prepared(req.plan, "moe.decode")
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=s.device)
                for s in st.scratch.scratch_specs()
            )
            binding = fused_moe.bind(
                req.plan, a=x, experts=experts, topk_ids=ids, topk_weights=w,
                output=out, scratch=scratch, input_scales_static=True,
            )
            run = lambda: fused_moe.run(binding=binding)  # noqa: E731
            out.fill_(float("nan"))
            run()
            torch.cuda.synchronize()
            eager = digest(out)
            out.fill_(float("nan"))
            compiled = torch.compile(run, fullgraph=True)
            compiled()
            torch.cuda.synchronize()
            traced = digest(out)
            words = sum(s.numel() for s in scratch)
            result.close()
            return eager, traced, words
        finally:
            session.close()
            for name in knobs:
                if saved[name] is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = saved[name]

    root = "/storage/models/Kimi-K3.1-Pro-EXL3-2bit"
    man = read_exl3_manifest(root)
    g = man.geometry
    M, H, K, E_ = 20, 3584, 16, 896   # the J-G 20-row shape (320 routes)
    layer = read_exl3_layer(root, man, 40, first_slot=0, slot_count=12)
    source, weights = trellis_from_exl3(layer)
    plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity="situ", io_dtype=torch.bfloat16,
            rotation_dtype=torch.float16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=g.num_experts, hidden_size=g.hidden_size,
            intermediate_size=384,
        ),
    )
    experts = fused_moe.prepare_weights(plan=plan, weights=weights, device="cuda")
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(M, H, generator=gen) * 0.0625).bfloat16().cuda()
    ids = torch.stack(
        [torch.randperm(E_, generator=gen)[:K] for _ in range(M)]
    ).to(device="cuda", dtype=torch.int32)
    w = torch.softmax(torch.randn(M, K, generator=gen), -1).cuda()
    out = torch.empty_like(x)

    off_eager, off_traced, off_words = build_and_run(False)
    on_eager, on_traced, on_words = build_and_run(True)
    assert off_eager == off_traced, "knob-off compiled run differs from eager"
    assert on_eager == on_traced, "knob-on compiled run differs from eager"
    assert on_eager == off_eager, "knob-on output differs from knob-off (not bit-exact)"
    assert on_words > off_words and (on_words - off_words) % W4A16_FLOW_WORKSPACE_EXTRA == 0, (
        f"flow knob did not extend the planned workspace "
        f"({off_words} -> {on_words}, want a positive multiple of "
        f"+{W4A16_FLOW_WORKSPACE_EXTRA} words)"
    )
    return (
        "compiled == eager both knob states; output bit-equal across states; "
        f"flow workspace engaged ({off_words} -> {on_words} words)"
    )


CASES = {
    "t17-compaction": case_t17_compaction,
    "w23-route-pack-flow": case_w23_route_pack_flow,
    # future graph-touching riders add their case function here.
}

failed = 0
for name, fn in CASES.items():
    try:
        detail = fn()
        print(f"dynamo-rider-smoke: {name}: OK ({detail})")
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"dynamo-rider-smoke: {name}: FAIL {type(exc).__name__}: {exc}")
if failed:
    raise SystemExit(1)
print(f"dynamo-rider-smoke: OK — {len(CASES)} rider case(s) trace, run and match eager")
