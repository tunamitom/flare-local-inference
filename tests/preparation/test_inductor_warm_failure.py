"""A rank whose warm execution fails raises instead of waiting in a failure gather."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.inductor_tuning import InductorSelections


class _Forward:
    owner = SimpleNamespace(compilation_variant="shared")

    def __init__(self, failing_call):
        self.calls, self.failing_call = 0, failing_call

    def __call__(self):
        self.calls += 1
        if self.calls == self.failing_call:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 32.00 MiB.")


@pytest.mark.parametrize("failing_call", [1, 2])
def test_failed_execution_propagates_without_gathering(monkeypatch, failing_call):
    def forbidden(*args, **kwargs):
        raise AssertionError("warm failure regression initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    selections = InductorSelections("cuda:0", coordinator=SimpleNamespace(ranks=[0, 1, 2, 3], group=None))
    gathered = []

    def gather(value):
        gathered.append(value)
        return [value] * 4

    monkeypatch.setattr(selections, "_gather", gather)
    bound = _Forward(failing_call)
    selections.forwards[id(bound)] = SimpleNamespace(_tuning_frozen=False), None, {}
    # Peers that enqueued the forward wait on its collectives and never reach a failure gather.
    with pytest.raises(torch.cuda.OutOfMemoryError, match="32.00 MiB"):
        selections.warm(bound, nullcontext)
    reconciliation = [[], None, None] if failing_call == 2 else []
    assert gathered == [("shared", []), *reconciliation]
