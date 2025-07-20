"""Reward functions for audio tasks."""

# %%
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from jiwer import process_words
import jiwer.transforms as tr
from whisper_normalizer.english import EnglishTextNormalizer


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

    def __post_init__(self):
        """Ensure that the values are non-negative."""
        assert self.n_hit >= 0 and self.n_err >= 0 and self.n_ref >= 0, f"Match error: {self}"

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

    def get_result_string(self):
        """Get the result string."""
        return f"Accuracy: {self.accuracy:.2f}%, Error Rate: {self.error_rate:.2f}%, Error Count: {self.count_err}/{self.n_ref}"


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
            # tr.ReduceToListOfListOfWords(),
        ]
    )
    norm = EnglishTextNormalizer()
    ref = norm(ref)
    hyp = norm(hyp)
    output = process_words(ref, hyp)
    hit = output.hits
    total = output.hits + output.substitutions + output.deletions
    miss = output.substitutions + output.deletions + output.insertions
    return Match(hit, miss, total)


def unbias_match(ref, hyp):
    """Compute the unbias match for a list of completions."""
    return bias_match(ref, hyp, bias=False)


def tagged_pieces(text):
    """Extract keywords from the text based on biasing words."""
    tagged_pieces = re.findall(r"\*.*?\*", text)
    pieces = [piece.strip("*").strip() for piece in tagged_pieces]
    return pieces


def bias_match(ref, hyp, bias=True):
    """Compute the bias match for a list of completions."""

    norm = EnglishTextNormalizer()
    pieces = [piece.strip() for piece in map(norm, tagged_pieces(ref)) if piece.strip()]
    # pieces can be a phrase with multiple words, so we split them into words
    words = [wd for piece in pieces for wd in piece.split() if wd.strip()]
    ref = norm(ref)
    hyp = norm(hyp)
    total = len(words) if bias else len(ref.split()) - len(words)
    if total == 0:
        return Match(0, 0, 0)
    hit = 0
    ins = 0
    for tag, ref_part, hyp_part in get_align(ref, hyp):
        if tag == "equal":
            bias_cnt = sum(len(piece.split()) for piece in pieces if piece in ref_part)
            hit += bias_cnt if bias else len(ref_part.split()) - bias_cnt
        elif tag == "insert":
            bias_cnt = sum(len(piece.split()) for piece in pieces if piece in hyp_part)
            ins += bias_cnt if bias else len(hyp_part.split()) - bias_cnt

    hit = min(hit, total)  # hit > total means that biasing piece has >2 words, partially matched. treat them as unmatched case, made #hit >#total
    return Match(hit, total - hit + ins, total)


def compute_match(groups, match_func=word_match, nbest=1):
    """compuate the overal match"""
    total = Match(0, 0, 0)
    for group in groups:
        best = Match(0, 100, 100)
        for sample in group[:nbest]:
            cur = match_func(sample["ref"], sample["hyp"])
            if cur.error_rate <= best.error_rate:
                best = cur
        total.n_err += best.n_err
        total.n_hit += best.n_hit
        total.n_ref += best.n_ref
    return total


def compute_biasing_metrics(groups):
    """compute biasing metrics"""
    # Extract reference and hypothesis pairs from groups

    return {
        "WER": compute_match(groups, word_match, 1).error_rate,
        "WER_A": compute_match(groups, word_match, 100).error_rate,
        "BWER": compute_match(groups, bias_match, 1).error_rate,
        "BWER_A": compute_match(groups, bias_match, 100).error_rate,
        "UWER": compute_match(groups, unbias_match, 1).error_rate,
        "UWER_A": compute_match(groups, unbias_match, 100).error_rate,
        "num_examples": len(groups),
    }


def eval_biasing_metrics(groups):
    """compute eval metrics"""
    # Extract reference and hypothesis pairs from groups
    extracted_groups = [[{"ref": egs["text"], "hyp": egs["completions"]} for egs in g] for g in groups]
    return compute_biasing_metrics(extracted_groups)


def reward_bias_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [bias_match(ref, completion).accuracy for completion, ref in zip(completions, references)]


def reward_word_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [word_match(ref, completion).accuracy for completion, ref in zip(completions, references)]


def reward_bias_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [-bias_match(ref, completion).count_err for completion, ref in zip(completions, references)]


def reward_word_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [-word_match(ref, completion).count_err for completion, ref in zip(completions, references)]


def reward_bias_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [-bias_match(ref, completion).error_rate for completion, ref in zip(completions, references)]


def reward_word_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [-word_match(ref, completion).error_rate for completion, ref in zip(completions, references)]


# %%

if __name__ == "__main__":
    pairs = [
        {
            "hyp": "Who was it she was in love with? The story will tell, I took upon myself to reply. Oh, I can't wait for the story. The story won't tell, said *douglas* Not in any literal, vulgar way. Nor is the pity then.",
            "ref": "who was it she was in love with the story will tell i took upon myself to reply oh i can't wait for the story the story won't tell said *douglas* not in any *literal* vulgar way *more's* the pity then",
        },
        {
            "hyp": "The air and the earth are curiously *mated* and *intermingled* as if the one were the breath of the other,",
            "ref": "the air and the earth are curiously *mated* and *intermingled* as if the one were the breath of the other",
        },
        {
            "hyp": "These thoughts agitated me all day, and my imagination scarcely *calmed* down after several hours' sleep.",
            "ref": "these thoughts agitated me all day and my imagination scarcely *calmed* down after several hours sleep",
        },
        {
            "hyp": "The task will not be difficult, returned David, hesitating, though I greatly fear your presence would rather increase than,*mitigate* his unhappy fortunes.",
            "ref": "the task will not be difficult returned david *hesitating* though i greatly fear your presence would rather increase than *mitigate* ,his unhappy fortunes",
        },
        {
            "hyp": "it was silent and gloomy, *beeing* *tenanted* *solely* by the *captive* and lighted by the dying *embers* of a fire which had been,used for the purpose of *cookery*",
            "ref": "it was silent and gloomy being *tenanted* *solely* by the *captive* and lighted by the dying *embers* of a fire which had been used ,for the *purposed* of *cookery*",
        },
        {
            "hyp": "or of the habits of our people it is quite impossible.",
            "ref": "or of the habits of our people it is quite impossible",
        },
        {
            "hyp": "To be or not to be, that is the question Whether 'tis *nobler* in the mind to suffer the *slings* and arrows what? No, *hamlet*,speaking",
            "ref": "to be or not to be that is the question whether tis *nobler* in the mind to suffer the *slings* and arrows what no *hamlet* ,speaking",
        },
        {
            "hyp": "By quick *marches* through these *inaccessible* mountains, that general *freed* himself from the superior forces of the,*covenanters*",
            "ref": "by quick *marches* through these *inaccessible* mountains that general *freed* himself from the superior forces of the ,*covenanters*",
        },
        {
            "hyp": "This *nobleman's* character, though celebrated for political courage and conduct, was very low for military *prowess* and after some,*skirmishes* in which he was *worsted* he here allowed *montrose* to escape him.",
            "ref": "this *nobleman's* character though celebrated for political courage and conduct was very low for military *prowess* and after some ,*skirmishes* in which he was *worsted* he here allowed *montrose* to escape him",
        },
    ]

    for pair in pairs:
        print("Bias match:", bias_match(pair["ref"], pair["hyp"]).error_rate)
        print("Unbias match:", unbias_match(pair["ref"], pair["hyp"]).error_rate)
        print("Word match:", word_match(pair["ref"], pair["hyp"]).error_rate)
        print()

# %
# %%
