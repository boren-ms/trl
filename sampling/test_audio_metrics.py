
# %%
from trl.scripts.audio_metrics import compute_wers

results = [
    {
        "ref": "If *he'd* run out of turnip seed he *wouldn't* *dress up* and *take* the buggy to go for more. ",
        "hyp": "If he'd run out of turnip seed he would not dress up and take the buggy to go for more.",
    }
]


for i, result in enumerate(results):
    wer, u_wer, b_wer = compute_wers([{**result, "id": f"{i}"}])
    print(f"Result {i}:")
    print("WER:", wer.get_result_string())  # noqa
    print("U-WER:", u_wer.get_result_string())  # noqa
    print("B-WER:", b_wer.get_result_string())  # noqa

# %%
