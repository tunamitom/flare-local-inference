# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 stage 0: cross-stream ordering of the decode shared-expert overlap (CPU, fake streams)."""

import pytest

torch = pytest.importorskip("torch")

from tests.recipes.test_hy4_w6_cp import tiny_available, worlds  # noqa: E402,F401


# --------------------------------------------------------------------------
# consult18 stage 0: the shared-expert side stream waits for its inputs
# --------------------------------------------------------------------------
def test_shared_expert_side_stream_waits_for_the_main_stream(worlds, monkeypatch):
    """The decode-graph overlap launches the shared expert on a side stream.
    It must first wait for the main stream (which wrote its inputs) and the
    main stream must join it before the add; the result equals the serial form."""
    import contextlib
    import types

    block = next(layer.mlp for layer in worlds[1][0].weights.layers
                 if hasattr(layer.mlp, "router"))
    x = torch.randn(3, worlds[1][0].model_config.hidden)
    serial = block.forward(x, overlap_shared=False)
    events = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait_stream", self.name, other.name))

        def wait_event(self, event):
            events.append(("wait_event", self.name, event.stream))

    class FakeEvent:
        stream = None

        def record(self, stream):
            self.stream = stream.name
            events.append(("record", stream.name))

    main = FakeStream("main")

    @contextlib.contextmanager
    def on(stream):
        events.append(("enter", stream.name))
        yield
        events.append(("exit", stream.name))

    original = block._shared_forward

    def shared_forward(*args):
        events.append(("shared_forward",))
        return original(*args)

    monkeypatch.setattr(torch.cuda, "Stream", lambda: FakeStream("side"))
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", on)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: main)
    monkeypatch.setattr(block, "_shared_forward", shared_forward)
    monkeypatch.setattr(block, "device", types.SimpleNamespace(type="cuda"))
    monkeypatch.setattr(block, "_shared_stream", None, raising=False)
    overlapped = block.forward(x, overlap_shared=True)
    assert events.index(("wait_stream", "side", "main")) < events.index(("enter", "side")) \
        < events.index(("shared_forward",)) < events.index(("record", "side"))
    assert events[-1] == ("wait_event", "main", "side"), "the add must wait for the side stream"
    assert torch.equal(overlapped, serial)
