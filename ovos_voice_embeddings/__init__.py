import os
from typing import Optional, Union

import numpy as np
from ovos_chromadb_embeddings import ChromaEmbeddingsDB
from ovos_config.locations import get_xdg_cache_save_path
from ovos_plugin_manager.templates.embeddings import VoiceEmbeddingsStore, EmbeddingsDB
from ovos_utils.log import LOG
from resemblyzer import VoiceEncoder, preprocess_wav
from speech_recognition import Recognizer, AudioFile, AudioData


class VoiceEmbeddingsRecognitionPlugin(VoiceEmbeddingsStore):
    def __init__(self, db: Optional[Union[EmbeddingsDB, str]] = None):
        if db is None:
            db_path = get_xdg_cache_save_path("chromadb")
            os.makedirs(db_path, exist_ok=True)
            db = f"{db_path}/resemblyzer_voice_prints"
        if isinstance(db, str):
            LOG.info(f"Using chromadb as voice embeddings store: {db}")
            db = ChromaEmbeddingsDB(db)
        super().__init__(db)
        self.encoder = VoiceEncoder()

    def get_voice_embeddings(self, audio_data: np.ndarray) -> np.ndarray:
        """audio data from a OVOS microphone"""
        if isinstance(audio_data, AudioData):
            audio_data = audio.get_wav_data()
        if isinstance(audio_data, bytes):
            audio_data = self.audiochunk2array(audio_data)
        return self.encoder.embed_utterance(audio_data)


if __name__ == "__main__":
    # Example usage:
    v = VoiceEmbeddingsRecognitionPlugin()

    a = "/home/miro/PycharmProjects/ovos-user-id/2609-156975-0001.flac"
    b = "/home/miro/PycharmProjects/ovos-user-id/qCCWXoCURKY.mp3"
    b2 = "/home/miro/PycharmProjects/ovos-user-id/4glfwiMXgwQ.mp3"

    with AudioFile(a) as source:
        audio = Recognizer().record(source)
    v.add_voice("user", audio)

    wav = preprocess_wav(b)
    v.add_voice("donald", wav)

    wav = preprocess_wav(b2)
    print(v.predict(wav))
    print(v.query(wav))
