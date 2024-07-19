# VoiceEmbeddingsRecognitionPlugin

The `VoiceEmbeddingsRecognitionPlugin` is a plugin for recognizing and managing voice embeddings. It uses the `VoiceEmbeddingsRecognizer` class from the `ovos_plugin_manager` template and integrates with the `ChromaEmbeddingsDB` for storing and retrieving voice embeddings. This plugin also leverages the `resemblyzer` library to generate voice embeddings and the `speech_recognition` library to handle audio data.

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

```

