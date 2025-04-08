from datasets import load_dataset
import fsspec

# Set a longer timeout (e.g., 60 seconds)
fsspec.config.conf['timeout'] = 600
cache_dir = "/datablob1/users/boren/data/librispeech_asr"
data = load_dataset("openslr/librispeech_asr", trust_remote_code=True,  cache_dir=cache_dir, num_proc=8)