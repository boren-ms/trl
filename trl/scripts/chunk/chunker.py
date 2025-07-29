""" chunk audio into segments """

# pylint: disable=redefined-builtin,relative-beyond-top-level,logging-fstring-interpolation
import logging
from whisper import load_audio
from .svad import SVadChunker
from .webrtc_vad import WebRtcChunker
import soundfile as sf
# from webrtc_vad import webrtc_segment
LOG = logging.getLogger(__name__)


class FixedWindowChunker:
    """chunk audio into segments with a non-overlapping sliding window of fixed length"""

    def __init__(self, max_len_sec=30):
        self.max_len_sec = max_len_sec

    def chunk(self, audio, sr):
        """chunk audio into segments"""
        max_len = self.max_len_sec * sr
        segments = []
        start = 0
        while start < len(audio):
            end = min(start + max_len, len(audio))
            segments.append((start / sr, end / sr))
            start = end
        return segments


class Chunker:
    """a chunker class to wrap different chunking methods"""

    def __init__(self, max_len=None, type="svad", **args):
        self.max_len = max_len
        self.type = type.lower()

        self._chunker = None
        if type == "none" or max_len is None:
            pass
        elif type == "fixed":
            self._chunker = FixedWindowChunker(max_len_sec=max_len)
        elif type == "svad":
            self._chunker = SVadChunker(max_len_sec=max_len, **args)
        elif type == "webrtc":
            self._chunker = WebRtcChunker(max_len, mode=args.get("chunk_mode", 0))
        else:
            raise ValueError(f"unknown chunker type: {type}")

    def __call__(self, audio_file):
        """chunk audio into segments"""
        segments = self._chunk(audio_file)
        seg_str = ", ".join([f"{s[0]:.2f}-{s[1]:.2f}" for s in segments])
        # print(f"chunked {len(segments)} segments: {seg_str}")
        LOG.info(f"chunked {len(segments)} segments: {seg_str}")

        large_seg_str = ", ".join([f"{s:.2f}-{e:.2f}" for s, e in segments if e - s > 30])
        if large_seg_str:
            # print(f"large segments: {large_seg_str}")
            LOG.info(f"large segments: {large_seg_str}")
        return segments

    def _chunk(self, audio_file):
        """chunk audio into segments"""
        if isinstance(self._chunker, WebRtcChunker):
            return self._chunker.chunk(audio_file)
        # the load_audio function will try to upsample the audio to 16k by default.
        # so that the sr is always 16k
        info = sf.info(audio_file)
        if self._chunker is None:
            return [(0, info.duration)]
        audio, sr = load_audio(audio_file), 16000
        return self._chunker.chunk(audio, sr)
