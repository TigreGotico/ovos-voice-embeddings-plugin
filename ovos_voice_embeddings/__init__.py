from typing import Any, Dict, Optional, Union

import numpy as np
from ovos_plugin_manager.templates.embeddings import EmbeddingsArray, VoiceEmbedder
from speakeronnx import SpeakerEmbedder

from ovos_voice_embeddings.version import __version__

Audio = Union[np.ndarray, bytes, bytearray]


class SpeakerOnnxVoiceEmbedder(VoiceEmbedder):
    """Speaker embeddings from a speakeronnx model, onnxruntime only.

    ``get_embeddings`` accepts either a float waveform at the ``sample_rate``
    config key (default 16000) or raw signed 16-bit little-endian PCM bytes at
    that rate, and returns the model's L2-normalised utterance embedding
    (256 dimensions for the default ``wespeaker-resnet34``).

    Config keys:
        model: speakeronnx model alias or path to an ONNX file (default wespeaker-resnet34)
        sample_rate: sample rate of the input audio (default 16000)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sample_rate = int(self.config.get("sample_rate", 16000))
        self.embedder = SpeakerEmbedder(model=self.config.get("model", "wespeaker-resnet34"))

    def get_embeddings(self, audio_data: Audio) -> EmbeddingsArray:
        if isinstance(audio_data, (bytes, bytearray)):
            audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio = np.asarray(audio_data, dtype=np.float32)
        target_sr = self.embedder.sample_rate
        if self.sample_rate != target_sr:
            n = int(round(len(audio) * target_sr / self.sample_rate))
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        return np.asarray(self.embedder.embed(audio), dtype=np.float32)
