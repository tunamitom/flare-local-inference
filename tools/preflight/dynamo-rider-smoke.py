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


CASES = {
    "t17-compaction": case_t17_compaction,
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
