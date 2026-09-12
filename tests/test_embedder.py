from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ovos_voice_embeddings import SpeakerOnnxVoiceEmbedder

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> np.ndarray:
    audio, sr = sf.read(FIXTURES / name, dtype="float32")
    assert sr == 16000
    return audio


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def embedder() -> SpeakerOnnxVoiceEmbedder:
    return SpeakerOnnxVoiceEmbedder()


@pytest.fixture(scope="module")
def embeddings(embedder):
    return {n: embedder.get_embeddings(load(f"{n}.wav")) for n in ("speaker_a_1", "speaker_a_2", "speaker_b_1")}


def test_embedding_shape_and_norm(embeddings):
    for emb in embeddings.values():
        assert emb.shape == (256,)
        assert emb.dtype == np.float32
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-3


def test_same_speaker_closer_than_different(embeddings):
    same = cosine(embeddings["speaker_a_1"], embeddings["speaker_a_2"])
    cross_1 = cosine(embeddings["speaker_a_1"], embeddings["speaker_b_1"])
    cross_2 = cosine(embeddings["speaker_a_2"], embeddings["speaker_b_1"])
    assert same > cross_1 + 0.1, (same, cross_1)
    assert same > cross_2 + 0.1, (same, cross_2)


def test_pcm_bytes_match_float_input(embedder, embeddings):
    audio = load("speaker_a_1.wav")
    pcm = (audio * 32767).astype(np.int16).tobytes()
    from_bytes = embedder.get_embeddings(pcm)
    assert cosine(from_bytes, embeddings["speaker_a_1"]) > 0.999


def test_embedding_is_deterministic(embedder, embeddings):
    again = embedder.get_embeddings(load("speaker_a_2.wav"))
    assert np.allclose(again, embeddings["speaker_a_2"], atol=1e-5)


def test_plugin_is_discoverable_by_opm():
    from ovos_plugin_manager.embeddings import find_voice_embeddings_plugins

    plugins = find_voice_embeddings_plugins()
    assert plugins["ovos-voice-embeddings-plugin"] is SpeakerOnnxVoiceEmbedder
