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
        self.tokens_to_remove = [
            x for x in self.tokens_to_remove if x not in self.exclude
        ]
        # print(f"tokens_to_remove: {self.tokens_to_remove}")



def get_align(reference, hypothesis, btag="*"):
    """Aligns the reference and hypothesis strings and returns the alignment details."""
    refs = reference.split()
    hyps = hypothesis.split()
    matcher = SequenceMatcher(
        None, [x.strip(btag) for x in refs], [x.strip(btag) for x in hyps]
    )
    alignment = []
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        alignment.append((operation, " ".join(refs[i1:i2]), " ".join(hyps[j1:j2])))
    return alignment


def count_tagged(text):
    """Count the number of tagged phrases in the text."""
    tagged = re.findall(r"\*.*?\*", text)
    return len([x for x in tagged if len(x) >= 1])


@dataclass
class Match:
    """Class to hold the match number."""

    n_hit: int
    n_err: int
    n_ref: int


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

def bias_match(ref, hyp):
    """Compute the reward for a list of completions."""
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
    total = count_tagged(ref)
    if total == 0:
        return Match(0, 0, 0)
    match = 0
    for tag, ref_part, _ in get_align(ref, hyp):
        if tag == "equal":
            match += count_tagged(ref_part)

    return Match(match, total - match, total)


def accuracy(match: Match):
    """Compute the accuracy."""
    if match.n_ref == 0:
        return 0
    return match.n_hit / match.n_ref * 100


def count_err(match: Match):
    """Compute the error count."""
    if match.n_ref == 0:
        return 0
    return match.n_err


def error_rate(match: Match):
    """Compute the error count."""
    if match.n_ref == 0:
        return 0
    return match.n_err / match.n_ref * 100


def reward_bias_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [
        accuracy(bias_match(ref, completion[-1]["content"]))
        for completion, ref in zip(completions, references)
    ]


def reward_word_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [
        accuracy(word_match(ref, completion[-1]["content"]))
        for completion, ref in zip(completions, references)
    ]


def reward_bias_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [
        -count_err(bias_match(ref, completion[-1]["content"]))
        for completion, ref in zip(completions, references)
    ]


def reward_word_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [
        -count_err(word_match(ref, completion[-1]["content"]))
        for completion, ref in zip(completions, references)
    ]


def reward_bias_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [
        -error_rate(bias_match(ref, completion[-1]["content"]))
        for completion, ref in zip(completions, references)
    ]


def reward_word_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [
        -error_rate(word_match(ref, completion[-1]["content"]))
        for completion, ref in zip(completions, references)
    ]


# %%

if __name__ == "__main__":
    hyp_text = "have you not met it *anywhere*"
    ref_text = "*have* *you* *not* *met them* *anywhere*"
    print("Bias error:", bias_match(ref_text, hyp_text))
    print("Word error:", word_match(ref_text, hyp_text))

# %%
