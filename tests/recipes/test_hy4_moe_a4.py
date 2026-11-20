"""consult18 E2: the MoE calibration registry and the A4 sidecar conversion."""
import torch


def test_calibration_registry_reset_and_save(tmp_path, monkeypatch):
    from ignition.recipes.hy4.moe import MoeCalibration

    calibration = MoeCalibration()
    monkeypatch.setenv("HY4_MOE_CALIBRATE", "1")
    assert MoeCalibration.enabled()
    buffer = calibration.buffer("model.layers.1.mlp.experts.fused", 4, torch.device("cpu"))
    assert buffer.shape == (1, 4, 2) and calibration.buffer(
        "model.layers.1.mlp.experts.fused", 4, torch.device("cpu")) is buffer
    buffer.fill_(3.0)
    path = tmp_path / "amax" / "r0.pt"
    assert calibration.save(str(path)) == 1
    saved = torch.load(path)
    assert torch.equal(saved["model.layers.1.mlp.experts.fused"], torch.full((4, 2), 3.0))
    calibration.reset()
    assert not buffer.any()
    monkeypatch.setenv("HY4_MOE_CALIBRATE", "0")
    assert not MoeCalibration.enabled()


def test_a4_scales_are_multipliers_with_the_widest_range_for_unrouted_experts(tmp_path, monkeypatch):
    from ignition.recipes.hy4 import moe

    amax = torch.tensor([[2.0, 10.0], [0.0, 0.0], [4.0, 20.0]])
    torch.save({"model.layers.3.mlp.experts.fused": amax}, tmp_path / "r5.pt")
    monkeypatch.setenv("HY4_MOE_A4", str(tmp_path))
    moe._A4_SIDECARS.clear()
    inputs, intermediates = moe.a4_activation_scales("model.layers.3.mlp.experts", rank=5)
    gscale = 6.0 * 448.0
    assert torch.allclose(inputs, torch.tensor([gscale / 2.0, gscale / 4.0, gscale / 4.0]))
    assert torch.allclose(intermediates, torch.tensor([gscale / 10.0, gscale / 20.0, gscale / 20.0]))
    monkeypatch.delenv("HY4_MOE_A4")
    assert moe.a4_activation_scales("model.layers.3.mlp.experts", rank=5) is None


def test_runtime_ab_switches(tmp_path):
    import json

    import pytest

    from ignition.recipes.hy4 import runtime

    path = tmp_path / "ab.json"
    try:
        _check_switches(runtime, path)
    finally:
        runtime.load(str(tmp_path / "absent.json"))                 # always restore the defaults


def _check_switches(runtime, path):
    import json

    import pytest

    assert runtime.load(str(path)) == runtime.DEFAULTS              # no file: defaults
    assert runtime.DEFAULTS["index_scores"] == "fp32"
    path.write_text(json.dumps({"index_scores": "bf16"}))
    assert runtime.load(str(path))["index_scores"] == "bf16"
    assert runtime.SWITCHES["index_scores"] == "bf16"
    path.write_text(json.dumps({"index_scores": "bf16", "ihc": "tuned"}))
    loaded = runtime.load(str(path))
    assert (loaded["index_scores"], loaded["ihc"]) == ("bf16", "tuned")
    path.write_text(json.dumps({"index_scores": "fp16"}))
    with pytest.raises(ValueError):
        runtime.load(str(path))
    path.unlink()
    assert runtime.load(str(path))["index_scores"] == "fp32"


def test_begin_forward_host_matches_the_device_path():
    """consult18 P1/P2: host-side bounds/ownership give the device path's
    `(kept, slots)` for every rank, start and width — decode (width 1) on a
    rank that does not own the token included (kept is empty there)."""
    from ignition.recipes.hy4.decoder import Hy4Cache

    class Cache:
        owner = Hy4Cache.owner
        local = Hy4Cache.local
        slots_for = Hy4Cache.slots_for
        _slots = Hy4Cache._slots
        _owned_slots = Hy4Cache._owned_slots
        begin_forward_host = Hy4Cache.begin_forward_host
        _extents = Hy4Cache._extents
        base_of = Hy4Cache.base_of
        cap_of = Hy4Cache.cap_of
        row_base_dev = Hy4Cache.row_base_dev
        row_cap_dev = Hy4Cache.row_cap_dev
        row_base = None
        row_cap = None

        def __init__(self, rank, world=8, capacity=4096, rows=2):
            self.rank, self.world, self.capacity, self.rows = rank, world, capacity, rows
            self.local_capacity = capacity // world
            self.device = torch.device("cpu")
            self._forward_memo = {}

    for rank in range(8):
        for row in (0, 1):
            for start, width in ((0, 1), (5, 1), (7, 1), (1000, 1), (8, 64), (13, 57), (100, 1024),
                                 (4000, 96)):
                cache = Cache(rank)
                tokens = torch.full((width,), row, dtype=torch.long)
                positions = torch.arange(start, start + width, dtype=torch.long)
                want_kept, want_slots = cache._owned_slots(tokens, positions)
                cache.begin_forward_host(tokens, positions, row=row, start=start)
                got_kept, got_slots = cache._owned_slots(tokens, positions)
                assert torch.equal(got_kept, want_kept), (rank, row, start, width)
                assert torch.equal(got_slots, want_slots), (rank, row, start, width)
