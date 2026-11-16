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
