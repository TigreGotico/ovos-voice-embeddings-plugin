# VoiceEmbeddingsRecognitionPlugin

The `VoiceEmbeddingsRecognitionPlugin` is a plugin for recognizing and managing voice embeddings.

It uses [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) to extract speaker embeddings and integrates with  [ovos-chromadb-embeddings-plugin](https://github.com/TigreGotico/ovos-chromadb-embeddings-plugin) for storing and retrieving voice embeddings. 

## Features

- **Voice Embeddings Extraction**: Converts audio data into voice embeddings using the `VoiceEncoder` from `resemblyzer`.
- **Voice Data Storage**: Stores and retrieves voice embeddings using `ChromaEmbeddingsDB`.
- **Voice Data Management**: Allows for adding, querying, and predicting voice embeddings associated with user IDs.
- **Supports Multiple Audio Formats**: Can handle audio data in various formats, including `wav` and `flac`.

## Usage

Here is a quick example of how to use the `VoiceEmbeddingsRecognitionPlugin`:

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

