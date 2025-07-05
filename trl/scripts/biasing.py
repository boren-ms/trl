# %%
import re
import random
from collections import deque
import blobfile as bf


def text_norm(word, tn_prob=1.0):
    """Normalize the text by removing special characters and converting to lowercase."""
    if random.random() <= tn_prob:
        word = " ".join(re.findall(r"[\w'\-]+", word))
        word = word.lower()
    return word


def split_text(text, max_len=1):
    """Split the text into pieces of max_len words."""
    words = text.split()
    if max_len <= 1:
        return words
    pieces = []
    while words:
        n = random.randint(1, max_len)
        pieces.append(" ".join(words[:n]))
        words = words[n:]
    return pieces


def tag_piece(piece, tag="*"):
    """Tag the piece with a specified tag."""
    return f"{tag}{piece}{tag}"

def tag_pieces(pieces, tag="*", specified=None, norm=None):
    """Tag the pieces with a specified tag."""
    if specified is None:
        return [tag_piece(p, tag) for p in pieces]
    norm = norm if norm is not None else lambda x: x
    specified = {norm(p) for p in specified}
    return [tag_piece(p, tag) if norm(p) in specified else p for p in pieces]

def rand_sample(lst, max_num, new=False):
    """Randomly sample random num <max_num elements from the list."""
    if new:
        n = min(max_num, len(lst))
        n = random.randint(0, n)
    else:
        n = random.randint(0, max_num)
        n = min(n, len(lst))
    # print(f"Sampling {n} pieces from {len(lst)}[{max_num}]")
    return random.sample(lst, n)


def read_words(file_path, N=None, tn_prob=1.0):
    """Read the top N lines from a file."""
    words = []
    with bf.BlobFile(file_path, "r") as f:
        for i, line in enumerate(f):
            if N is not None and i >= N:
                break
            word = line.split()[0]
            words.append(text_norm(word, tn_prob))
    return words


class PieceSampler:
    """Sample segments from the text using instance parameters."""

    def __init__(
        self,
        buffer_size=100000,
        bias_prob=1.0,
        hit_prob=0.5,
        max_piece_len=1,
        new_sampling=False,
        common_word_file=None,
        common_word_num=None,
        max_num=10,
        tag="*",
        tag_all=True,
        log_interval=None,
    ):
        """Initialize the PieceSampler with configuration parameters.

        Args:
            buffer_size: Maximum size of the buffer for negative pieces
            max_len: Maximum words of text piece
            bias_prob: biasing prompt probability
            max_num: Maximum number of pieces to sample
            hit_prob: Probability of sampling from positive words
        """
        self.buffer = deque(maxlen=buffer_size)
        self.max_piece_len = max_piece_len
        self.bias_prob = bias_prob
        self.max_num = max_num
        self.hit_prob = hit_prob
        self.tag = tag
        self.tag_all = tag_all
        self.new_sampling = new_sampling
        self.log_interval = int(log_interval) if log_interval else None
        self.common_words = set(read_words(common_word_file, common_word_num) if common_word_file else [])
        self.idx = 0

    def filter_commons(self, examples):
        """Filter out common words from the examples."""
        if not self.common_words:
            return examples
        new_examples = []
        for phrase in examples:
            if all(text_norm(wd) in self.common_words for wd in phrase.split()):
                continue
            new_examples.append(phrase)
        return new_examples


    def _sample(self, pieces):
        """Sample segments from the positive pieces."""
        self.buffer.extend(pieces)
        if random.random() > self.bias_prob:
            return []
        num_pieces = random.randint(1, self.max_num)
        examples = []
        if random.random() <= self.hit_prob:
            examples = rand_sample(pieces, num_pieces, new=self.new_sampling)
        examples += rand_sample(self.buffer, num_pieces - len(examples), new=self.new_sampling)
        if self.common_words:
            examples = [p for p in examples if text_norm(p) not in self.common_words]
        return self.filter_commons(examples)

    def sample(self, text):
        """Sample segments from the text using instance parameters."""
        pieces = split_text(text, self.max_piece_len)
        examples = self._sample(pieces)
        examples = list(set(examples))
        random.shuffle(examples)
        # Tag the pieces with shared tags
        shared = set(examples) & set(pieces)
        pieces = tag_pieces(pieces, self.tag, shared)
        if self.tag_all:  # tag the input sample pieces as well
            examples = tag_pieces(examples, self.tag)
        self.idx += 1
        prompt, trans = ", ".join(examples), " ".join(pieces)
        if self.log_interval is not None and self.idx % self.log_interval == 0:
            print(f"[{self.idx}] biasing  list: {prompt}")
            print(f"[{self.idx}] transcription: {trans}")

        return prompt, trans


# %%
if __name__ == "__main__":
    sample_utterances = [
        "Hello, how are you?",
        "This is a test sentence.",
        "I love programming in Python!",
        "The quick brown fox jumps over the lazy dog.",
        "Data science is an interdisciplinary field.",
        "Artificial intelligence is transforming many industries.",
        "Learning to code is a valuable skill in today's job market.",
        "The meeting is scheduled for tomorrow at 9 AM.",
        "She sold seashells by the seashore.",
        "Machine learning models require large amounts of data.",
        "Please remember to submit your assignment by Friday.",
        "The restaurant on Main Street serves delicious pasta.",
        "Climate change is affecting ecosystems worldwide.",
        "Thank you for your help with this project.",
        "The concert was canceled due to bad weather.",
        "Neural networks are inspired by the human brain.",
        "Remember to back up your files regularly.",
        "The new movie received excellent reviews from critics.",
        "Quantum computing has the potential to revolutionize technology.",
        "The library closes at 8 PM on weekdays.",
        "Self-driving cars use various sensors to navigate.",
        "Regular exercise is important for maintaining good health.",
        "The art exhibition features works from local artists.",
        "Natural language processing helps computers understand human language.",
        "Don't forget to water the plants while I'm away.",
    ]

    # Create an instance of PieceSampler
    sampler = PieceSampler(buffer_size=100, max_piece_len=3, bias_prob=1, max_num=100, hit_prob=0.9, log_interval=2)
    # Sample pieces from the utterances
    for text in sample_utterances:
        examples, trans = sampler.sample(text)
        print(f"Original Trans: {text}")
        print(f"Updated  Trans: {trans}")
        print(f"Sampled examples: {examples}")
        print("Buffer:", len(sampler.buffer))
        print("-" * 40)
    # # %%

# %%
