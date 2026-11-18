"""Origin-keyed export evidence: an MSE build keeps the census bar; a GPTQ build proves its
trained codes by a calibration objective below the base package's (consult 10 section 1)."""
import json
import os
import time
from types import SimpleNamespace

import pytest

from ignition.recipes.hy4 import weights
from ignition.recipes.hy4.weights import (
    EXPORT_MAX_RELATIVE_ERROR, GPTQ_CENSUS_CEILING, _gptq_evidence_problems, verify_export_package)

CONFIG = SimpleNamespace(layers=3, experts=2, draft_layers=1)
SHA = "a" * 64


def _objective(gptq=1.0, base=2.0):
    return {f"model.layers.{layer}.mlp.experts.{expert}.{projection}": [gptq, base]
            for layer in (1, 2) for expert in range(2) for projection in ("gate_up_proj", "down_proj")}


def _evidence(**changes):
    evidence = {"base_report_sha256": SHA, "calibration_sha256": SHA, "objective": _objective()}
    evidence.update(changes)
    return evidence


def _package(tmp_path, *, origin=None, census_max, evidence=None):
    prefixes = ["model.layers.1.mlp.experts", "model.layers.2.mlp.experts", "model.mtp_layers.0.mlp.experts"]
    expected = (CONFIG.layers - 1 + CONFIG.draft_layers) * CONFIG.experts * 3
    report = {
        "byte_verified": True,
        "packed_matrices": expected,
        "census": {f"m{index}": 0.08 for index in range(expected)},
        "census_max_relative": census_max,
        "window_edge_counts": {prefix: {projection: [0] * CONFIG.experts
                                        for projection in ("gate_proj", "up_proj", "down_proj")}
                               for prefix in prefixes},
    }
    if origin is not None:
        report["producer"] = {"origin": origin}
    if evidence is not None:
        report["gptq_evidence"] = evidence
    (tmp_path / "config.json").write_text("{}")
    shard = tmp_path / "model-00001.safetensors"
    shard.write_bytes(b"")
    os.chmod(shard, 0o644)
    (tmp_path / "export_report.json").write_text(json.dumps(report))
    time.sleep(0.01)
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    return tmp_path


def test_gptq_evidence_holds():
    assert _gptq_evidence_problems(_evidence(), CONFIG) == []


@pytest.mark.parametrize("evidence, needle", [
    (None, "carries no gptq_evidence"),
    (_evidence(base_report_sha256=None), "base_report_sha256"),
    (_evidence(calibration_sha256="short"), "calibration_sha256"),
    (_evidence(objective={"x": [1.0, 2.0]}), "covers 1 of 8"),
    (_evidence(objective=_objective(gptq=2.0, base=2.0)), "8 matrices"),
    (_evidence(objective=_objective(gptq=0.0, base=0.0)), "not strictly below"),
])
def test_gptq_evidence_refusals(evidence, needle):
    problems = _gptq_evidence_problems(evidence, CONFIG)
    assert any(needle in problem for problem in problems), problems


def test_mse_package_keeps_the_census_bar(tmp_path):
    verify_export_package(_package(tmp_path, census_max=0.0829), CONFIG)
    with pytest.raises(ValueError, match="census_max_relative"):
        verify_export_package(_package(tmp_path, census_max=EXPORT_MAX_RELATIVE_ERROR + 0.01), CONFIG)


def test_gptq_census_is_not_an_mse_package(tmp_path):
    # trained codes declared as an MSE build stay refused by the census bar
    with pytest.raises(ValueError, match="census_max_relative"):
        verify_export_package(_package(tmp_path, origin="ptq-mse", census_max=0.11, evidence=_evidence()), CONFIG)


def test_gptq_package_loads_with_evidence(tmp_path):
    verify_export_package(_package(tmp_path, origin="gptq", census_max=0.11, evidence=_evidence()), CONFIG)


def test_gptq_package_refusals(tmp_path):
    with pytest.raises(ValueError, match="gptq_evidence"):
        verify_export_package(_package(tmp_path, origin="gptq", census_max=0.11), CONFIG)
    with pytest.raises(ValueError, match="census_max_relative"):
        verify_export_package(_package(tmp_path, origin="gptq", census_max=GPTQ_CENSUS_CEILING + 0.01,
                                       evidence=_evidence()), CONFIG)
    with pytest.raises(ValueError, match="unknown origin"):
        verify_export_package(_package(tmp_path, origin="awq", census_max=0.08), CONFIG)
    assert weights.EXPORT_ORIGINS == ("ptq-mse", "gptq")
