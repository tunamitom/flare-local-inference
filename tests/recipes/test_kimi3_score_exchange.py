# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""J-missed §2 (upstream 8b1be5b): the candidate-score exchange as a plan setting.

Wiring (Plan field ``score_exchange``, default armed, dropped with ``--no-score-exchange``;
never an environment switch) and the channel contract: armed, the bootstrap does not
allocate the (score, position) pair channel and the selections channel carries decode and
verification candidate scores at 2 * TOPK BF16 columns per rank (8,192 bytes per row and
rank); dropped, the pair channel returns and the wiring is the pre-port one. CPU only; the
J-fault battery, the wide-round content cell and the temperature-0 comparison ride the
later GPU wave (levers/j-missed-fable-review.md §2).
"""
import ast
from pathlib import Path

from ignition import planning
from ignition.cli import _parser

ROOT = Path(planning.__file__).resolve().parent / "recipes" / "kimi3"


def test_the_plan_setting_is_authoritative():
    # The plan setting flows Plan -> recipe -> PreparationConfig -> PreparedDecoder ->
    # PreparedMLA -> PreparedDSA, with no environment read anywhere on the path.
    for name in ("bootstrap.py", "decoder.py", "attention.py", "dsa.py"):
        source = (ROOT / name).read_text()
        assert "SCORE_EXCHANGE" not in source
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    assert PreparationConfig().score_exchange is True
    assert PreparationConfig(score_exchange=False).score_exchange is False
    assert planning.Plan.__dataclass_fields__["score_exchange"].default is True


def test_the_cli_flag_is_the_only_switch():
    dropped = _parser().parse_args(["export", "kimi-k3", "--no-score-exchange"])
    assert dropped.no_score_exchange is True
    armed = _parser().parse_args(["export", "kimi-k3"])
    assert armed.no_score_exchange is False
    source = (ROOT.parent.parent / "planning.py").read_text()
    assert "score_exchange: bool = True" in source
    assert '"score_exchange": self.score_exchange' in source


def test_the_bootstrap_channels_follow_the_setting():
    source = (ROOT / "bootstrap.py").read_text()
    tree = ast.parse(source)
    # Armed: the pair channel is not allocated and the selections channel covers the decode
    # and verification rows; dropped: the pair channel returns at 4 * TOPK columns per rank.
    assert "if sparse and not config.score_exchange else None" in source
    assert "max_rows=max(selection_rows, prefill_selection_rows) if config.score_exchange" in source
    assert "columns=4 * TOPK * shard.world_size" in source
    assert "columns=2 * TOPK * shard.world_size" in source
    # The setting reaches the decoders through the shared collectives dict (also speculation's).
    collectives = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "collectives" for t in n.targets)]
    assert collectives and any("score_exchange=config.score_exchange" in ast.unparse(n.value)
                               for n in collectives)


def test_the_dsa_stage_selects_the_form_by_the_setting():
    source = (ROOT / "dsa.py").read_text()
    # The pair form remains for a rank that attends over gathered context; decode and
    # verification rows take the share form when armed and the pair form when dropped.
    assert "PreparedScoreExchangeTopK(local, scores, selected," in source
    assert 'runtime=context.selections' in source and 'name=stage_name + ".candidate_scores"' in source
    assert 'name=stage_name + ".candidates"' in source
    assert "self.shards.world > 1 and self.gathered_context" in source
    assert "context.candidates is None" in source
