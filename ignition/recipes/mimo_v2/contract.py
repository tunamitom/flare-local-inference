"""Prepared graph inputs, continuation storage, and frontend syntax."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p

from .audio import AUDIO_CAPACITIES
from .vision import FEATURES, PATCH_CAPACITIES


VOCABULARY = 152576
INPUT_VOCABULARY = 152576
VISION = {'features_per_request': FEATURES, 'max_images': 8, 'patches_per_feature': 4,
          'patch_capacities': list(PATCH_CAPACITIES)}
AUDIO = {'frame_capacities': list(AUDIO_CAPACITIES)}
AUDIO_FEATURE_TOKEN = 151669
# Video needs no graph of its own: its frames are vision patches. The sampling and resize
# settings live in the package's frontend block (preparation/frontend.py video_settings); here
# the recipe only names the tokens the prompt builder splices around them.
VIDEO_TOKENS = dict(feature_token=151656, start_token=151670, end_token=151671)
FRONTEND = c.frontend(tools=p.QWEN_CODER, terminal_markers=["<tool_call>"],
    assistant_marker="<|im_start|>assistant\n", message_end_markers=["<|im_end|>"], rejected_tokens=[],
    prefix_boundary=c.prefix_boundary(["system", "developer"], append_role="user", minimum_messages=0,
        required_start="<|im_start|>system\n", required_marker="<|im_end|>"),
    image_tokens=dict(feature_token=151655, span=None),
    image_preparation=p.MIMO_IMAGE_PREPARATION, image_layout=p.MIMO_IMAGE_LAYOUT,
    audio_preparation=p.MIMO_AUDIO_PREPARATION, audio_tokens=dict(feature_token=AUDIO_FEATURE_TOKEN),
    video_tokens=VIDEO_TOKENS)


def prepare(signatures, buffers, pool, *, image_slots=4, page_tokens, draft_lag, rows=1, embedding_tables=(), regions=()):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    if any(s.kind == "audio" for s in signatures.values()):
        inputs["audio"] = c.audio_inputs({field for s in signatures.values() if s.kind == "audio" for field in s.inputs})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    pending = c.state_view("request_pending_token", 8)
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending,
        cursor=cursor, rows=rows, embedding_tables=embedding_tables)
    return c.model_contract(vocabulary=VOCABULARY, input_vocabulary=INPUT_VOCABULARY,
        page_tokens=page_tokens, page_tables=[dict(buffer="page_tables", page_offset=0),
            *([dict(buffer="draft_page_tables", page_offset=1)] if "draft_page_tables" in buffers else [])],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=graphs, vision={**VISION, "feature_slots": image_slots}, frontend=FRONTEND, audio=AUDIO)
