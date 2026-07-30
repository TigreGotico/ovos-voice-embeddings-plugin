# VoiceEmbeddingsRecognitionPlugin

`VoiceEmbeddingsRecognitionPlugin` recognizes and manages voice embeddings for OVOS. It uses [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) to extract speaker embeddings from audio, and it stores and retrieves those embeddings with [ovos-chromadb-embeddings-plugin](https://github.com/TigreGotico/ovos-chromadb-embeddings-plugin).

## Features

- **Voice embeddings extraction**: converts audio data into voice embeddings with the `VoiceEncoder` from `resemblyzer`.
- **Voice data storage**: stores and retrieves voice embeddings with `ChromaEmbeddingsDB`.
- **Voice data management**: adds, queries, and predicts voice embeddings for user IDs.
- **Multiple audio formats**: handles audio data in formats that include `wav` and `flac`.

## Install

```bash
pip install ovos-voice-embeddings-plugin
```

## Usage

This example enrolls two voices, then identifies a third audio sample against them.

```python
from ovos_voice_embeddings import VoiceEmbeddingsRecognitionPlugin
from resemblyzer import preprocess_wav
from speech_recognition import Recognizer, AudioFile
from ovos_chromadb_embeddings import ChromaEmbeddingsDB

db = ChromaEmbeddingsDB("./voice_db")
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
print(v.prompt(wav))

```

## Related projects

- [ovos-chromadb-embeddings-plugin](https://github.com/TigreGotico/ovos-chromadb-embeddings-plugin) — the ChromaDB-backed embeddings store this plugin uses.
- [ovos-user-id](https://github.com/TigreGotico/ovos-user-id) — multi-user identity layer for OVOS. It uses this plugin as its voice recognizer for authentication.

## License

MIT.
