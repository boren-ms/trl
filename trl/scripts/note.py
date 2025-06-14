# %%

import soundfile as sf
import blobfile as bf
import blobpath as bp

# %%

# file_path = "az://orngwus2cresco/data/boren/data/LibriSpeech/train-other-500/1653/142374/1653-142374-0024.flac"
file_path = "/home/boren/data/LibriSpeech/train-other-500/1653/142374/1653-142374-0024.flac"


print(bf.exists(file_path))
#%%
with bf.BlobFile(file_path, "rb") as f:
    audio, fs = sf.read(f, dtype="float32")
    print(f"Audio shape: {audio.shape}, Sample rate: {fs}")
    # Process the audio data as needed
    # For example, you can print the first few samples
    print("First few samples:", audio[:10])
#%%    
az_path = bp.BlobPath(file_path)
audio, fs = sf.read(az_path.open("rb"), dtype="float32")



# %%
