# ovos-voice-embeddings-plugin

An OVOS voice embeddings plugin. It turns a speech clip into a speaker
embedding with [speakeronnx](https://github.com/TigreGotico/speakeronnx), so
the runtime needs `onnxruntime` and `numpy` only, no PyTorch. The default model
is `wespeaker-resnet34`, a 256-dimension L2-normalised embedding downloaded from
the [OpenVoiceOS speaker-embeddings-onnx](https://huggingface.co/collections/OpenVoiceOS/speaker-embeddings-onnx)
collection on first use. Two clips of one speaker give vectors with a high
cosine similarity. Clips of different speakers give a low one.

The plugin implements the `VoiceEmbedder` template of
[ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager) and
registers under the `opm.embeddings.voice` entry point as
`ovos-voice-embeddings-plugin`.

## Install

```bash
pip install ovos-voice-embeddings-plugin
```

## Usage

```python
import numpy as np
import soundfile as sf
from ovos_voice_embeddings import SpeakerOnnxVoiceEmbedder

embedder = SpeakerOnnxVoiceEmbedder()          # wespeaker-resnet34, 16 kHz input

alice_1, _ = sf.read("alice_1.wav", dtype="float32")
alice_2, _ = sf.read("alice_2.wav", dtype="float32")
bob, _ = sf.read("bob.wav", dtype="float32")

a1 = embedder.get_embeddings(alice_1)
a2 = embedder.get_embeddings(alice_2)
b = embedder.get_embeddings(bob)

print(a1.shape)                  # (256,)
print(float(np.dot(a1, a2)))     # same speaker, about 0.8
print(float(np.dot(a1, b)))      # different speakers, about 0.4
```

The embeddings are L2-normalised, so the dot product is the cosine similarity.
`get_embeddings` also accepts raw signed 16-bit PCM bytes, which is what the
OVOS listener hands to plugins.

### Store and match voices

Pair the embedder with any OVOS embeddings database plugin, for example
[ovos-chromadb-embeddings-plugin](https://github.com/TigreGotico/ovos-chromadb-embeddings-plugin):

```python
from ovos_chromadb_embeddings import ChromaEmbeddingsDB

db = ChromaEmbeddingsDB({"path": "./voice_db"})
db.add_embeddings("alice", a1)
db.add_embeddings("bob", b)

print(db.query(a2, top_k=1))     # [("alice", distance)]
```

## Configuration

| key | default | description |
|---|---|---|
| `model` | `"wespeaker-resnet34"` | speakeronnx model alias or path to an ONNX file |
| `sample_rate` | `16000` | sample rate of the audio passed to `get_embeddings` |

## Tests

```bash
pip install -e ".[test]"
pytest tests
```

The tests embed three real clips from LibriSpeech test-clean (CC BY 4.0) and
assert the vector shape and that two clips of one speaker are closer than clips
of two speakers.

## License

Apache-2.0.
