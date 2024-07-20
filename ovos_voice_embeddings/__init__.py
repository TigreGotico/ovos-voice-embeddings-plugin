import numpy as np
from ovos_plugin_manager.templates.embeddings import VoiceEmbeddingsStore, EmbeddingsDB
from resemblyzer import VoiceEncoder, preprocess_wav
from speech_recognition import Recognizer, AudioFile, AudioData


class VoiceEmbeddingsRecognitionPlugin(VoiceEmbeddingsStore):
    def __init__(self, db: EmbeddingsDB, thresh: float = 0.75):
        super().__init__(db, thresh)
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
    from ovos_chromadb_embeddings import ChromaEmbeddingsDB
    path = "/tmp/voice_db"
    db = ChromaEmbeddingsDB(path)
    v = VoiceEmbeddingsRecognitionPlugin(db)

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
