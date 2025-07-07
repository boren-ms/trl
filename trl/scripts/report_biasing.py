# %%
import re
from collections import deque
from enum import Enum
import logging
import json
from pathlib import Path
import fire
import wandb
from whisper_normalizer.english import EnglishTextNormalizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


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
        assert self.ref_words != 0
        errors = self.errors[Code.substitution] + self.errors[Code.insertion] + self.errors[Code.deletion]
        return 100.0 * errors / self.ref_words

    def get_result_string(self):
        return (
            f"error_rate={self.get_wer()}, "
            f"ref_words={self.ref_words}, "
            f"subs={self.errors[Code.substitution]}, "
            f"ins={self.errors[Code.insertion]}, "
            f"dels={self.errors[Code.deletion]}"
        )


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
                best_score = self.scores_[i - 1][j - 1] + (
                    self.cost(ref, hyp, Code.match) if ref == hyp else self.cost(ref, hyp, Code.substitution)
                )

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


def load_tsv(ref_file, hyp_file):
    """Load reference and hypothesis files in TSV format."""
    refs = {}
    with open(ref_file, "r", encoding="utf8") as f:
        for line in f:
            ary = line.strip().split("\t")
            uttid, ref, biasing_words = ary[0], ary[1], set(json.loads(ary[2]))
            refs[uttid] = {"text": ref, "biasing_words": biasing_words}

    hyps = {}
    with open(hyp_file, "r", encoding="utf8") as f:
        for line in f:
            ary = line.strip().split("\t")
            # May have empty hypo
            if len(ary) >= 2:
                uttid, hyp = ary[0], ary[1]
            else:
                uttid, hyp = ary[0], ""
            hyps[uttid] = hyp
    return refs, hyps


def load_json(ref_file, hyp_file):
    """Load reference and hypothesis files in JSON format."""
    hyps = {}
    with open(hyp_file, "r", encoding="utf8") as f:
        data = json.load(f)[:-1]
        for utt in data:
            hyps[utt["audio_ids"][0]] = utt["generated_texts"][0]
    refs = {}
    with open(ref_file, "r", encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            refs[data["id"]] = {
                "text": data["text"],
                "biasing_words": set(data["ground_truth"]),
            }
    return refs, hyps


def load_file(ref_file, hyp_file):
    """Load reference and hypothesis files based on their file type."""
    if ref_file.endswith(".tsv"):
        return load_tsv(ref_file, hyp_file)
    elif ref_file.endswith(".jsonl"):
        return load_json(ref_file, hyp_file)
    else:
        raise ValueError("Unsupported file type. Please provide a .tsv or .json file.")


def calc_wers(refs, hyps):
    """Calculate WER, U-WER, and B-WER."""
    wer = WordError()
    u_wer = WordError()
    b_wer = WordError()
    norm = EnglishTextNormalizer()
    # norm = lambda x: x
    for uttid, ref in refs.items():
        if uttid not in hyps:
            continue
        ref_tokens = norm(ref["text"]).split()
        biasing_words = ref["biasing_words"]
        hyp_tokens = norm(hyps[uttid]).split()
        ed = EditDistance()
        result = ed.align(ref_tokens, hyp_tokens)
        for code, ref_idx, hyp_idx in zip(result.codes, result.refs, result.hyps):
            if code == Code.match:
                wer.ref_words += 1
                if ref_tokens[ref_idx] in biasing_words:
                    b_wer.ref_words += 1
                else:
                    u_wer.ref_words += 1
            elif code == Code.substitution:
                wer.ref_words += 1
                wer.errors[Code.substitution] += 1
                if ref_tokens[ref_idx] in biasing_words:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.substitution] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.substitution] += 1
            elif code == Code.deletion:
                wer.ref_words += 1
                wer.errors[Code.deletion] += 1
                if ref_tokens[ref_idx] in biasing_words:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.deletion] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.deletion] += 1
            elif code == Code.insertion:
                wer.errors[Code.insertion] += 1
                if hyp_tokens[hyp_idx] in biasing_words:
                    b_wer.errors[Code.insertion] += 1
                else:
                    u_wer.errors[Code.insertion] += 1
    return wer, u_wer, b_wer


def write_score(output_file, wer):
    """Write WER scores to a file."""
    scores = [
        {"details": wer.get_result_string()},
        {"final_score": f"WER: {wer.get_wer()}"},
    ]
    logger.info("Scores written to %s", output_file)
    with open(output_file, "w", encoding="utf8") as f:
        json.dump(scores, f, indent=4)


def extract_int(x, v=-1):
    """Extract an integer from a string or return a default value."""
    if isinstance(x, int):
        return x
    match = re.search(r"(\d+)", str(x))
    return int(match.group(1)) if match else v


def main(ref_file, hyp_file, output=False, exp_name=None, chkp_num=None, dataset=None):
    """Main function to compute WER, U-WER, and B-WER."""
    refs, hyps = load_file(ref_file, hyp_file)
    logger.info("Loaded %d reference utts from %s", len(refs), ref_file)
    logger.info("Loaded %d hypothesis utts from %s", len(hyps), hyp_file)

    miss = set(refs.keys()) - set(hyps.keys())
    if miss:
        logger.warning("Missing IDs in hypothesis: %s", miss)

    wer, uwer, bwer = calc_wers(refs, hyps)
    print(f"WER: {wer.get_result_string()}")
    print(f"U-WER: {uwer.get_result_string()}")
    print(f"B-WER: {bwer.get_result_string()}")
    if output:
        pfx_path = str(Path(hyp_file).with_suffix(""))
        write_score(f"{pfx_path}_wer.txt", wer)
        write_score(f"{pfx_path}_uwer.txt", uwer)
        write_score(f"{pfx_path}_bwer.txt", bwer)
    if exp_name is not None:
        print(f"Logging to wandb for {exp_name}@{chkp_num}")
        dataset = dataset or "default"
        chkp = extract_int(chkp_num, -1)
        with wandb.init(project="ls_biasing", name=exp_name, id=exp_name, resume="allow") as run:
            wandb.log(
                {
                    f"{dataset}/wer": wer.get_wer(),
                    f"{dataset}/uwer": uwer.get_wer(),
                    f"{dataset}/bwer": bwer.get_wer(),
                    "chkp": chkp,
                }
            )
            wandb.define_metric("*", step_metric="chkp")


if __name__ == "__main__":
    fire.Fire(main)
