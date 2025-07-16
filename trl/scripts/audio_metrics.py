# %%
import re
from collections import deque
from enum import Enum
from whisper_normalizer.english import EnglishTextNormalizer

class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4


class AlignmentResult(object):
    def __init__(self, refs, hyps, codes, score):
        self.refs = refs  # deque<int>
        self.hyps = hyps  # deque<int>
        self.codes = codes  # deque<Code>
        self.score = score  # float


class WordError(object):
    def __init__(self):
        self.errors = {
            Code.substitution: 0,
            Code.insertion: 0,
            Code.deletion: 0,
        }
        self.ref_words = 0

    def get_wer(self):
        if self.ref_words == 0:
            return 0.0
        # assert self.ref_words != 0
        # errors = self.errors[Code.substitution] + self.errors[Code.insertion] + self.errors[Code.deletion]
        return 100.0 * self.error_count / self.ref_words

    @property
    def accuracy(self):
        return 100.0 - self.get_wer()

    @property
    def error_count(self):
        return self.errors[Code.substitution] + self.errors[Code.insertion] + self.errors[Code.deletion]

    @property
    def error_rate(self):
        return self.get_wer() / 100.0

    def get_result_string(self):
        return f"error_rate={self.get_wer()}, " f"ref_words={self.ref_words}, " f"subs={self.errors[Code.substitution]}, " f"ins={self.errors[Code.insertion]}, " f"dels={self.errors[Code.deletion]}"


def coordinate_to_offset(row, col, ncols):
    return int(row * ncols + col)


def offset_to_row(offset, ncols):
    return int(offset / ncols)


def offset_to_col(offset, ncols):
    return int(offset % ncols)


class EditDistance(object):
    def __init__(self):
        self.scores_ = None
        self.backtraces_ = None
        self.confusion_pairs_ = {}
        self.inserted_words_ = {}
        self.deleted_words_ = {}

    def cost(self, ref, hyp, code):
        if code == Code.match:
            return 0
        elif code == Code.insertion or code == Code.deletion:
            return 3
        else:  # substitution
            return 4

    def get_result(self, refs, hyps):
        res = AlignmentResult(refs=deque(), hyps=deque(), codes=deque(), score=None)

        num_rows, num_cols = len(self.scores_), len(self.scores_[0])
        res.score = self.scores_[num_rows - 1][num_cols - 1]

        curr_offset = coordinate_to_offset(num_rows - 1, num_cols - 1, num_cols)

        while curr_offset != 0:
            curr_row = offset_to_row(curr_offset, num_cols)
            curr_col = offset_to_col(curr_offset, num_cols)

            prev_offset = self.backtraces_[curr_row][curr_col]

            prev_row = offset_to_row(prev_offset, num_cols)
            prev_col = offset_to_col(prev_offset, num_cols)

            res.refs.appendleft(curr_row - 1)
            res.hyps.appendleft(curr_col - 1)
            if curr_row - 1 == prev_row and curr_col == prev_col:
                ref_str = refs[res.refs[0]]
                deleted_word = ref_str
                if deleted_word not in self.deleted_words_:
                    self.deleted_words_[deleted_word] = 1
                else:
                    self.deleted_words_[deleted_word] += 1

                res.codes.appendleft(Code.deletion)

            elif curr_row == prev_row and curr_col - 1 == prev_col:
                hyp_str = hyps[res.hyps[0]]
                inserted_word = hyp_str
                if inserted_word not in self.inserted_words_:
                    self.inserted_words_[inserted_word] = 1
                else:
                    self.inserted_words_[inserted_word] += 1

                res.codes.appendleft(Code.insertion)

            else:
                # assert(curr_row - 1 == prev_row and curr_col - 1 == prev_col)
                ref_str = refs[res.refs[0]]
                hyp_str = hyps[res.hyps[0]]

                if ref_str == hyp_str:
                    res.codes.appendleft(Code.match)
                else:
                    res.codes.appendleft(Code.substitution)

                    confusion_pair = "%s -> %s" % (ref_str, hyp_str)
                    if confusion_pair not in self.confusion_pairs_:
                        self.confusion_pairs_[confusion_pair] = 1
                    else:
                        self.confusion_pairs_[confusion_pair] += 1

            curr_offset = prev_offset

        return res

    def align(self, refs, hyps):
        if len(refs) == 0 and len(hyps) == 0:
            raise ValueError("Doesn't support empty ref AND hyp!")

        # NOTE: we're not resetting the values in these matrices because every value
        # will be overridden in the loop below. If this assumption doesn't hold,
        # be sure to set all entries in self.scores_ and self.backtraces_ to 0.
        self.scores_ = [[0.0] * (len(hyps) + 1) for _ in range(len(refs) + 1)]
        self.backtraces_ = [[0] * (len(hyps) + 1) for _ in range(len(refs) + 1)]

        num_rows, num_cols = len(self.scores_), len(self.scores_[0])

        for i in range(num_rows):
            for j in range(num_cols):
                if i == 0 and j == 0:
                    self.scores_[i][j] = 0.0
                    self.backtraces_[i][j] = 0
                    continue

                if i == 0:
                    self.scores_[i][j] = self.scores_[i][j - 1] + self.cost(None, hyps[j - 1], Code.insertion)
                    self.backtraces_[i][j] = coordinate_to_offset(i, j - 1, num_cols)
                    continue

                if j == 0:
                    self.scores_[i][j] = self.scores_[i - 1][j] + self.cost(refs[i - 1], None, Code.deletion)
                    self.backtraces_[i][j] = coordinate_to_offset(i - 1, j, num_cols)
                    continue

                # Below here both i and j are greater than 0
                ref = refs[i - 1]
                hyp = hyps[j - 1]
                best_score = self.scores_[i - 1][j - 1] + (self.cost(ref, hyp, Code.match) if ref == hyp else self.cost(ref, hyp, Code.substitution))

                prev_row = i - 1
                prev_col = j - 1
                ins = self.scores_[i][j - 1] + self.cost(None, hyp, Code.insertion)
                if ins < best_score:
                    best_score = ins
                    prev_row = i
                    prev_col = j - 1

                delt = self.scores_[i - 1][j] + self.cost(ref, None, Code.deletion)
                if delt < best_score:
                    best_score = delt
                    prev_row = i - 1
                    prev_col = j

                self.scores_[i][j] = best_score
                self.backtraces_[i][j] = coordinate_to_offset(prev_row, prev_col, num_cols)

        return self.get_result(refs, hyps)


def text_norm(txt):
    """Normalize tokens by removing leading and trailing whitespace."""
    norm = EnglishTextNormalizer()
    if isinstance(txt, str):
        return norm(txt.strip())
    elif isinstance(txt, list):
        return [norm(x) for x in txt]
    else:
        raise ValueError(f"Unsupported type for text normalization: {type(txt)}. Expected str or list of str.")

def find_word_indices(text, pieces):
    indexs = []
    for piece in pieces:
        for m in re.finditer(re.escape(piece), text):
            start_idx = len(text[:m.start()].split()) # previous words
            piece_len = len(piece.split())
            indexs.extend(range(start_idx, start_idx + piece_len))
    return indexs

def calc_wers(refs, hyps):
    """Calculate WER, U-WER, and B-WER."""
    wer = WordError()
    u_wer = WordError()
    b_wer = WordError()
    for uttid, ref in refs.items():
        if uttid not in hyps:
            continue
        tn_ref = text_norm(ref["text"])
        tn_hyp = text_norm(hyps[uttid])
        bias_words = text_norm(ref["biasing_words"])
        bias_ref_indexs = find_word_indices(tn_ref, bias_words)
        bias_hyp_indexs = find_word_indices(tn_hyp, bias_words)

        ed = EditDistance()
        result = ed.align(tn_ref.split(), tn_hyp.split())
        for code, ref_idx, hyp_idx in zip(result.codes, result.refs, result.hyps):
            if code == Code.match:
                wer.ref_words += 1
                if ref_idx in bias_ref_indexs:
                    b_wer.ref_words += 1
                else:
                    u_wer.ref_words += 1
            elif code == Code.substitution:
                wer.ref_words += 1
                wer.errors[Code.substitution] += 1
                if ref_idx in bias_ref_indexs:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.substitution] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.substitution] += 1
            elif code == Code.deletion:
                wer.ref_words += 1
                wer.errors[Code.deletion] += 1
                if ref_idx in bias_ref_indexs:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.deletion] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.deletion] += 1
            elif code == Code.insertion:
                wer.errors[Code.insertion] += 1
                if hyp_idx in bias_hyp_indexs:
                    b_wer.errors[Code.insertion] += 1
                else:
                    u_wer.errors[Code.insertion] += 1
    return wer, u_wer, b_wer


def extract_keywords(text):
    """Extract keywords from the text based on biasing words."""
    tagged_words = re.findall(r"\*.*?\*", text)
    keywords = [wd.strip("*") for wd in tagged_words]
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # Remove tagged words from the text
    return {
        "biasing_words": keywords,
        "text": text,
    }


def compute_biasing_metrics(results):
    """compute biasing metrics"""
    wer, u_wer, b_wer = compute_wers(results)
    return {
        "WER": wer.get_wer(),
        "UWER": u_wer.get_wer(),
        "BWER": b_wer.get_wer(),
    }


def compute_wers(results):
    """compute WER, U-WER, and B-WER"""
    # Extract reference and hypothesis pairs from groups
    refs = {result["id"]: extract_keywords(result["ref"]) for result in results}
    hyps = {result["id"]: result["hyp"] for result in results}
    # Calculate WER, U-WER, and B-WER
    wer, u_wer, b_wer = calc_wers(refs, hyps)
    return wer, u_wer, b_wer


def eval_biasing_metrics(groups):
    """compute eval metrics"""
    # Extract reference and hypothesis from top group (i.e., the first group)
    results = [
        {
            "ref": group[0]["text"],
            "hyp": group[0]["completions"],
            "id": group[0].get("id", i),  # if "id" is not present, use index
        }
        for i, group in enumerate(groups)
    ]
    wer, u_wer, b_wer = compute_wers(results)
    return {
        "WER": wer.get_wer(),
        "UWER": u_wer.get_wer(),
        "BWER": b_wer.get_wer(),
    }


def compute_reward_wers(completions, **kwargs):
    """Compute rewards for a list of completions."""
    references = kwargs["text"]
    rewards = []
    for i, (completion, ref) in enumerate(zip(completions, references)):
        rewards.append(compute_wers([{"id": i, "ref": ref, "hyp": completion}]))
    return rewards


def reward_word_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [wer[0].accuracy for wer in wers]  # WER


def reward_unbias_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [wer[1].accuracy for wer in wers]  # U-WER


def reward_bias_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [wer[2].accuracy for wer in wers]  # B-WER


def reward_word_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[0].error_count for wer in wers]  # WER


def reward_unbias_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[1].error_count for wer in wers]  # U-WER


def reward_bias_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[2].error_count for wer in wers]  # B-WER


def reward_word_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[0].error_rate for wer in wers]  # WER


def reward_unbias_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[1].error_rate for wer in wers]  # U-WER


def reward_bias_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[2].error_rate for wer in wers]  # B-WER
