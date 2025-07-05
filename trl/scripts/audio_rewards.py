"""Reward functions for audio tasks."""

# %%
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from jiwer import process_words
import jiwer.transforms as tr



class RemovePunctuationExclude(tr.RemovePunctuation):
    """RemovePunctuation excluding certain characters."""

    def __init__(self, exclude=None):
        super().__init__()
        self.exclude = exclude or []
        self.tokens_to_remove = [x for x in self.tokens_to_remove if x not in self.exclude]
        # print(f"tokens_to_remove: {self.tokens_to_remove}")


def get_align(reference, hypothesis, btag="*"):
    """Aligns the reference and hypothesis strings and returns the alignment details."""
    refs = reference.split()
    hyps = hypothesis.split()
    matcher = SequenceMatcher(None, [x.strip(btag) for x in refs], [x.strip(btag) for x in hyps])
    alignment = []
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        alignment.append((operation, " ".join(refs[i1:i2]), " ".join(hyps[j1:j2])))
    return alignment


@dataclass
class Match:
    """Class to hold the match number."""

    n_hit: int
    n_err: int
    n_ref: int

    @property
    def accuracy(self):
        """Compute the accuracy."""
        if self.n_ref == 0:
            return 0
        return self.n_hit / self.n_ref * 100

    @property
    def count_err(self):
        """Compute the error count."""
        if self.n_ref == 0:
            return 0
        return self.n_err

    @property
    def error_rate(self):
        """Compute the error count."""
        if self.n_ref == 0:
            return 0
        return self.n_err / self.n_ref * 100


def word_match(ref, hyp):
    """Compute the word error rate between two strings."""
    norm = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.ExpandCommonEnglishContractions(),
            tr.RemovePunctuation(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(),
            tr.ReduceToListOfListOfWords(),
        ]
    )
    output = process_words(ref, hyp, norm, norm)
    hit = output.hits
    total = output.hits + output.substitutions + output.deletions
    miss = output.substitutions + output.deletions + output.insertions
    return Match(hit, miss, total)


def count_tagged(text, tagged=True):
    """Count the number of tagged phrases in the text."""
    tagged_piece = re.findall(r"\*.*?\*", text)
    n_tagged_word = sum([len(x.strip().split()) for x in tagged_piece])
    return n_tagged_word if tagged else len(text.split()) - n_tagged_word

def unbias_match(ref, hyp):
    """Compute the unbias match for a list of completions."""
    return bias_match(ref, hyp, bias=False)

def bias_match(ref, hyp, bias=True):
    """Compute the bias match for a list of completions."""
    norm = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.ExpandCommonEnglishContractions(),
            RemovePunctuationExclude(exclude=["*"]),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(),
        ]
    )
    ref = norm(ref)
    hyp = norm(hyp)
    total = count_tagged(ref, bias)
    if total == 0:
        return Match(0, 0, 0)
    hit = 0
    ins = 0
    for tag, ref_part, hyp_part in get_align(ref, hyp):
        if tag == "equal":
            hit += count_tagged(ref_part, bias)
        elif tag == "insert":
            ins += 0 if bias else len(hyp_part.strip().split())
    return Match(hit, total - hit + ins, total)


def compute_match(groups, match_func=word_match, nbest=1):
    """compuate the overal match"""
    total = Match(0, 0, 0)
    for group in groups:
        best = Match(0, 100, 100)
        for sample in group[:nbest]:
            cur = match_func(sample["text"], sample["completions"][-1]["content"])
            if cur.error_rate <= best.error_rate:
                best = cur
        total.n_err += best.n_err
        total.n_hit += best.n_hit
        total.n_ref += best.n_ref
    return total


def compute_metrics(groups):
    """compute eval metrics"""
    return {
        "WER": compute_match(groups, word_match, 1).error_rate,
        "WER_A": compute_match(groups, word_match, 100).error_rate,
        "BWER": compute_match(groups, bias_match, 1).error_rate,
        "BWER_A": compute_match(groups, bias_match, 100).error_rate,
        "UWER": compute_match(groups, unbias_match, 1).error_rate,
        "UWER_A": compute_match(groups, unbias_match, 100).error_rate,
        "num_egs": len(groups),
    }


def reward_bias_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [bias_match(ref, completion[-1]["content"]).accuracy for completion, ref in zip(completions, references)]


def reward_word_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [word_match(ref, completion[-1]["content"]).accuracy for completion, ref in zip(completions, references)]


def reward_bias_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [-bias_match(ref, completion[-1]["content"]).count_err for completion, ref in zip(completions, references)]


def reward_word_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [-word_match(ref, completion[-1]["content"]).count_err for completion, ref in zip(completions, references)]


def reward_bias_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [-bias_match(ref, completion[-1]["content"]).error_rate for completion, ref in zip(completions, references)]


def reward_word_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [-word_match(ref, completion[-1]["content"]).error_rate for completion, ref in zip(completions, references)]


# %%

if __name__ == "__main__":
    hyp_text = "have you hi not met it ,are you show"
    ref_text = "*have* *you* *not* *met them* *anywhere*"
    print("Bias error:", bias_match(ref_text, hyp_text))
    print("Word error:", word_match(ref_text, hyp_text))

# %%
