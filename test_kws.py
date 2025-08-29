# %%
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from torchaudio.sox_effects import apply_effects_file

effects = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]

# load a demo dataset and read audio files
# %%
model = AutoModelForAudioClassification.from_pretrained("Amirhossein75/Keyword-Spotting")
# %%
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")


# %%
from trl.scripts.audio_dataset import create_audio_dataset

ds = create_audio_dataset(
    "ls_bias",
    **{
        "num_egs": 10,
        "jsonl_path": "/home/boren/data/librispeech_biasing/ref/test-clean.biasing_100.jsonl",
    }
)


def map_to_array(example):
    speech, _ = apply_effects_file(example["audio_path"], effects)
    example["speech"] = speech.squeeze(0).numpy()
    return example


ds = ds.map(map_to_array)
print(ds[0])
# %%

# compute attention masks and normalize the waveform if needed
inputs = feature_extractor(ds[:4]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")

logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]

# %%
model.config.id2label
# %%
