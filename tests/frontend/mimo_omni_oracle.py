"""Run the vLLM fork's MiMo Omni processor as a parity oracle for native MiMo media handling.

The fork at ``FLARE_VLLM_KK`` (default ``/opt/vllm-kk``) ships ``MiMoOmniProcessor`` with
plain-PyTorch ports of torchaudio's ``MelSpectrogram`` and ``Resample``. This module loads
those two files without installing vLLM, stubbing the three vLLM utilities they import, and
reproduces the request-side media loading that precedes the processor: vLLM's uniform
``num_frames`` video sampling with one-second nominal timestamps, and channel-mean mono
audio resampled to 24 kHz with the torchaudio sinc kernel.
"""

import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

FORK = Path(os.environ.get("FLARE_VLLM_KK", "/opt/vllm-kk"))
PROCESSOR = FORK / "vllm/transformers_utils/processors/mimo_v2_omni.py"
AUDIO = FORK / "vllm/multimodal/audio.py"


def available():
    return PROCESSOR.is_file() and AUDIO.is_file()


def _stub(name, **attributes):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    """The fork's audio helpers and MiMo processor modules."""
    if "vllm.transformers_utils.processors.mimo_v2_omni" in sys.modules:
        return sys.modules["vllm.multimodal.audio"], sys.modules["vllm.transformers_utils.processors.mimo_v2_omni"]

    class PlaceholderModule:
        def __init__(self, name):
            self._name = name

        def placeholder_attr(self, attribute):
            return PlaceholderModule(f"{self._name}.{attribute}")

        def __getattr__(self, attribute):
            raise ImportError(f"{self._name} is not installed in the oracle environment")

    class _threads:
        def __init__(self, count):
            self.count = count

        def __enter__(self):
            self.previous = torch.get_num_threads()
            torch.set_num_threads(self.count)

        def __exit__(self, *exc):
            torch.set_num_threads(self.previous)

    _stub("vllm")
    _stub("vllm.utils")
    _stub("vllm.utils.import_utils", PlaceholderModule=PlaceholderModule)
    _stub("vllm.utils.torch_utils", set_default_torch_num_threads=_threads)
    _stub("vllm.multimodal")
    _stub("vllm.transformers_utils")
    _stub("vllm.transformers_utils.processors")
    audio = _load("vllm.multimodal.audio", AUDIO)
    processor = _load("vllm.transformers_utils.processors.mimo_v2_omni", PROCESSOR)
    return audio, processor


def hf_config(snapshot):
    config = json.loads((Path(snapshot) / "config.json").read_text())
    return SimpleNamespace(vision_config=config["vision_config"], processor_config=config["processor_config"],
                           audio_config=config["audio_config"], rope_scaling=None)


def processor(snapshot):
    from transformers import AutoTokenizer

    _, module = modules()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    return module.MiMoOmniProcessor.from_hf_config(tokenizer, hf_config(snapshot))


def sample_frame_indices(total_frames, fps, *, num_frames=32, target_fps=-1):
    """vLLM's ``VideoBackend.compute_frames_index_to_sample`` (OpenCV loader defaults)."""
    duration = total_frames / fps if fps > 0 else 0.0
    count = total_frames
    if num_frames > 0:
        count = min(num_frames, total_frames)
    if target_fps > 0:
        count = min(count, math.floor(duration * target_fps))
    count = max(1, count)
    if count == total_frames:
        return list(range(count))
    return np.linspace(0, total_frames - 1, count, dtype=int).tolist()


def video_item(frames):
    """vLLM hands MiMo's processor sampled RGB frames with one-second nominal timestamps."""
    frames = torch.as_tensor(np.asarray(frames)).permute(0, 3, 1, 2).float()
    return frames, torch.arange(frames.shape[0], dtype=torch.float32) / 1.0


def audio_item(waveform, sample_rate):
    """vLLM decodes at the native rate, averages channels, and resamples with the sinc kernel."""
    audio, _ = modules()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=0)
    if int(sample_rate) != 24000:
        waveform = audio.resample_sinc(torch.as_tensor(waveform), int(sample_rate), 24000).numpy()
    return torch.as_tensor(waveform, dtype=torch.float32), 24000
