"""
This is Silero-VAD's 16kHz model in PyTorch.
"""

from typing import Mapping, Optional, Tuple, Any

import torch
import torchaudio
from torch import Tensor


class SileroVADWrapper(torch.nn.Module):
    """Silero VAD model wrapper for 16kHz sample rate."""

    def __init__(self):
        super().__init__()
        self.model = SileroVAD()
        self.h: Optional[Tensor] = None
        self.c: Optional[Tensor] = None

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Load Silero VAD state dict."""
        return self.model.load_state_dict(state_dict, strict)

    def reset_states(self):
        """Reset hidden states."""
        self.h = None
        self.c = None

    def forward(self, audio: Tensor, sr: int) -> Tensor:
        """forward method for Silero VAD model"""
        assert sr == 16000
        out, self.h, self.c = self.model(audio, self.h, self.c)
        return out


# pylint: disable=invalid-name
class SileroVAD(torch.nn.Module):
    """Silero VAD model for 16kHz sample rate."""

    def __init__(self):
        super().__init__()
        channels = [258, 16, 32, 32, 64]
        dropout_rate = 0.15
        self.adaptive_normalization = AdaptiveAudioNormalizationNew()
        self.feature_extractor = STFT()
        self.first_layer = torch.nn.Sequential(
            ConvBlock(input_dim=channels[0], output_dim=channels[1]),
            torch.nn.Dropout(p=dropout_rate),
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=channels[1], out_channels=channels[1], kernel_size=1, stride=2),
            torch.nn.BatchNorm1d(num_features=channels[1]),
            torch.nn.ReLU(),
            torch.nn.Sequential(
                ConvBlock(input_dim=channels[1], output_dim=channels[2]),
                torch.nn.Dropout(p=dropout_rate),
            ),
            torch.nn.Conv1d(in_channels=channels[2], out_channels=channels[2], kernel_size=1, stride=2),
            torch.nn.BatchNorm1d(num_features=channels[2]),
            torch.nn.ReLU(),
            torch.nn.Sequential(
                ConvBlock(input_dim=channels[2], output_dim=channels[3], projection=False),
                torch.nn.Dropout(p=dropout_rate),
            ),
            torch.nn.Conv1d(in_channels=channels[3], out_channels=channels[3], kernel_size=1, stride=2),
            torch.nn.BatchNorm1d(num_features=channels[3]),
            torch.nn.ReLU(),
            torch.nn.Sequential(
                ConvBlock(input_dim=channels[3], output_dim=channels[4]),
                torch.nn.Dropout(p=dropout_rate),
            ),
            torch.nn.Conv1d(in_channels=channels[4], out_channels=channels[4], kernel_size=1, stride=1),
            torch.nn.BatchNorm1d(num_features=channels[4]),
            torch.nn.ReLU(),
        )
        self.decoder = VADDecoderRNNJIT()

    def forward(self, x: Tensor, h: Optional[Tensor] = None, c: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """forward method for Silero VAD model"""
        # x shape: [batch, 512 or 1024 or 1536]
        # x shape: [batch, 1536 * n] in batch mode
        _wsize = min(x.size(1), 1536)
        _n = _wsize // 512
        x0 = self.feature_extractor(x)  # shape: [batch, 129, 8 or 16 or 24]
        norm = self.adaptive_normalization(x0)  # shape: [batch, 129, 8 or 16 or 24]
        x1 = torch.cat([x0, norm], dim=1)  # shape: [batch, 258, 8 or 16 or 24]
        x2 = self.first_layer(x1)  # shape: [batch, 16, 8 or 16 or 24]
        x3 = self.encoder(x2)  # shape: [batch, 64, 1 or 2 or 3]
        (
            x4,
            h0,
            c0,
        ) = self.decoder(
            x3, h, c
        )  # x4 shape: [batch, 1, 1 or 2 or 3]
        # out = torch.mean(torch.squeeze(x4, dim=1), dim=1, keepdim=True)  # shape: [batch, 1]
        batch_size = x4.size(0)
        out = torch.mean(torch.squeeze(x4, dim=1).view(batch_size, -1, _n), dim=-1, keepdim=True)
        return (out, h0, c0)


class AdaptiveAudioNormalizationNew(torch.nn.Module):
    """Adaptive Audio Normalization"""

    def __init__(self):
        super().__init__()
        self.filter_ = torch.nn.Parameter(torch.zeros([1, 1, 7]), requires_grad=False)  # tensor([[[0.0366, 0.1113, 0.2167, 0.2707, 0.2167, 0.1113, 0.0366]]])
        self.to_pad = 3

    def forward(self, spect: Tensor) -> Tensor:
        """forward"""
        spect0 = torch.log1p(torch.mul(spect, 1048576))
        if len(spect0.size()) == 2:
            spect1 = torch.unsqueeze(spect0, 0)  # becomes 3D tensor
        else:
            spect1 = spect0  # shape: [batch, num_freq_bin, sequence]
        mean = torch.mean(spect1, dim=1, keepdim=True)  # avg over freq bins # shape: [batch, 1, sequence]
        mean0 = torch.nn.functional.pad(mean, [self.to_pad, self.to_pad], mode="reflect")  # shape: [batch, 1, sequence+to_pad+to_pad]
        mean1 = torch.nn.functional.conv1d(mean0, self.filter_)  # weighted sum over neighboring time frames, center higher, side lower
        mean_mean = torch.mean(mean1, dim=-1, keepdim=True)  # avg over time steps
        spect2 = spect1 - mean_mean
        return spect2


class STFT(torch.nn.Module):
    """STFT"""

    def __init__(self):
        super().__init__()
        self.filter_length = 256
        self.hop_length = 64
        self.forward_basis_buffer = torch.nn.Parameter(torch.zeros([258, 1, 256]), requires_grad=False)  # shape: [out_channel=258, in_channel=1, kernel_size=256]

    def forward(self, input_data: Tensor) -> Tensor:
        """forward"""
        return self.transform_(input_data)[0]  # only use magnitude

    def transform_(self, input_data: Tensor) -> Tuple[Tensor, Tensor]:
        """transform_"""
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        input_data = input_data.view([num_batches, 1, num_samples])
        to_pad = int((self.filter_length - self.hop_length) / 2)  # (256-64)/2=96
        input_data2 = torch.nn.functional.pad(input_data, [to_pad, to_pad], "reflect")  # shape: [B, in_channel=1, num_sample+to_pad+to_pad=512+96*2=704]
        forward_transform = torch.nn.functional.conv1d(
            input_data2,
            weight=self.forward_basis_buffer,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )  # [B, out_channel=258, output_len=(512+96*2-(256-1)-1)/64+1=8]
        cutoff = int(self.filter_length / 2 + 1)  # 256/2+1=129
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, -cutoff:, :]
        magnitude = torch.sqrt(torch.pow(real_part, 2) + torch.pow(imag_part, 2))
        phase = torch.atan2(imag_part, real_part)
        return (magnitude, phase)


class ConvBlock(torch.nn.Module):
    """ConvBlock"""

    def __init__(
        self,
        input_dim,
        output_dim,
        dw_conv_kernel_size=5,
        dw_conv_padding=2,
        pw_conv_kernel_size=1,
        projection=True,
        proj_kernel_size=1,
    ):
        super().__init__()
        self.dw_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=dw_conv_kernel_size,
                padding=dw_conv_padding,
                groups=input_dim,
            ),
            torch.nn.Identity(),
            torch.nn.ReLU(),
        )
        self.pw_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=pw_conv_kernel_size),
            torch.nn.Identity(),
        )
        self.projection = projection
        if self.projection:
            self.proj = torch.nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=proj_kernel_size)
        else:
            self.proj = None
        self.activation = torch.nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """forward"""
        x0 = self.pw_conv(self.dw_conv(x))
        if self.proj is not None:
            residual = self.proj(x)
            x1 = x0 + residual
        else:
            x1 = x0 + x
        return self.activation(x1)


class VADDecoderRNNJIT(torch.nn.Module):
    """VADDecoderRNNJIT"""

    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: Tensor, h: Optional[Tensor] = None, c: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x: shape [batch, 64, 1 or 2 or 3]
        """
        if h is not None and c is not None:
            x0, (h0, c0) = self.rnn(torch.permute(x, dims=[0, 2, 1]), (h, c))  # lstm input: [batch, seq_len, dim], when batch_first=True
        else:
            x0, (h0, c0) = self.rnn(torch.permute(x, dims=[0, 2, 1]), None)
        x3 = torch.permute(x0, [0, 2, 1])  # x: shape [batch, 64, 1 or 2 or 3]
        x4 = self.decoder(x3)
        return (x4, h0, c0)


def read_audio(path: str, sampling_rate: int = 16000):
    """read audio and convert to sampling rate"""
    sox_backends = {"sox", "sox_io"}
    audio_backends = torchaudio.list_audio_backends()

    if len(sox_backends.intersection(audio_backends)) > 0:
        effects = [["channels", "1"], ["rate", str(sampling_rate)]]

        wav, sr = torchaudio.sox_effects.apply_effects_file(path, effects=effects)
    else:
        wav, sr = torchaudio.load(path)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0)


@torch.no_grad()
def get_speech_probs(audio: torch.Tensor, vad_model: torch.nn.Module, window_size_samples: int = 512):
    """
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible
    model: preloaded pytorch silero VAD model
    window_size_samples: int (default - 512 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate
        and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model perfomance!!
    """
    audio_length_samples = len(audio)

    speech_probs = []
    h, c = None, None
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        chunk = chunk.unsqueeze(0)  # shape -> [batch_size=1, window_size_samples]
        speech_prob, h, c = vad_model(chunk, h, c)  # speech_prob shape: [batch_size=1, 1]
        speech_prob = speech_prob[0, 0].item()
        speech_probs.append(speech_prob)
    return speech_probs


def make_visualization(probs, step):
    """make visualization of speech probabilities"""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    pd.DataFrame({"probs": probs}, index=[x * step for x in range(len(probs))]).plot(
        figsize=(16, 8),
        kind="area",
        ylim=[0, 1.05],
        xlim=[0, len(probs) * step],
        xlabel="seconds",
        ylabel="speech probability",
        colormap="tab20",
    )


# pylint: disable=too-many-locals,too-many-branches,too-many-statements,duplicate-code
@torch.no_grad()
def get_speech_regions(
    audio: torch.Tensor,
    vad_model: torch.nn.Module,
    threshold: float = 0.5,
    sil_threshold: float = 0.35,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    visualize_probs: bool = False,
):
    """
    This method is used for splitting long audios into speech chunks using silero VAD

    Parameters
    ----------
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible

    model: preloaded pytorch silero VAD model

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this parameter
        for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

    sil_threshold: float (default - 0.35)
        Speech silence threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities BELOW this value are considered as SILENCE. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.35 is pretty good for most datasets.

    sampling_rate: int (default - 16000)
        Currently pytorch silero VAD models supports only 16000 sample rates

    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out

    max_speech_duration_s: int (default -  inf)
        Maximum duration of speech chunks in seconds
        Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence
        that lasts more than 100ms (if any), to prevent agressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.

    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it

    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate
        and 256, 512, 768 samples for 8000 sample rate. Values other than these may affect model
        perfomance!!

    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side

    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)

    visualize_probs: bool (default - False)
        whether draw prob hist or not

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks
        (samples or seconds based on return_seconds)
    """
    assert sampling_rate == 16000
    h, c = None, None
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        chunk = chunk.unsqueeze(0)  # shape -> [batch_size=1, window_size_samples]
        speech_prob, h, c = vad_model(chunk, h, c)  # speech_prob shape: [batch_size=1, 1]
        speech_prob = speech_prob[0, 0].item()
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    # to save potential segment end (and tolerate some silence)
    temp_end = 0
    # to save potential segment limits in case of maximum segment size reached
    prev_end = next_start = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if triggered and (window_size_samples * i) - current_speech["start"] > max_speech_samples:
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:  # previously reached silence (< neg_thres) and is still not speech (< thres)
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < sil_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if ((window_size_samples * i) - temp_end) > min_silence_samples_at_max_speech:
                # condition to avoid cutting in very short silence
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            current_speech["end"] = temp_end
            if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                speeches.append(current_speech)
            current_speech = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            continue

    if current_speech and (audio_length_samples - current_speech["start"]) > min_speech_samples:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - silence_duration // 2))
            else:
                speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - speech_pad_samples))
        else:
            speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))

    if return_seconds:
        for speech_dict in speeches:
            speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
            speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
    # elif step > 1:
    #     for speech_dict in speeches:
    #         speech_dict["start"] *= step
    #         speech_dict["end"] *= step

    if visualize_probs:
        make_visualization(speech_probs, window_size_samples / sampling_rate)

    return speeches


if __name__ == "__main__":
    import os
    from pathlib import Path

    silero_vad = SileroVAD()
    model_checkpoint = str(Path(__file__).parent / "resource/silero_vad_checkpoint.pt")
    silero_vad.load_state_dict(torch.load(model_checkpoint, map_location="cpu"), strict=False)
    # vad_model.decoder.rnn.flatten_parameters()
    # vad_model = torch.jit.script(vad_model)
    # vad_model = torch.compile(vad_model, mode="reduce-overhead")
    silero_vad.eval()
    silero_vad.cuda()

    SAMPLING_RATE = 16000
    test_audio = read_audio("model/audios/e18842e5-e5e3-e908-207e-59831d173b88_0.wav", sampling_rate=SAMPLING_RATE)
    test_audio = test_audio.cuda()
    audio_length = test_audio.shape[0] / SAMPLING_RATE
    print("audio length:", audio_length)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    audio_speech_probs = get_speech_probs(test_audio, silero_vad, window_size_samples=512)  # 512 samples for 16000 sample rate is 32ms
    torch.cuda.synchronize()
    start.record()
    for _ in range(1):
        audio_speech_probs = get_speech_probs(test_audio, silero_vad, window_size_samples=512)  # 512 samples for 16000 sample rate is 32ms
    end.record()
    torch.cuda.synchronize()
    # print("speech_probs for each 512 sample (32ms) frame", speech_probs)
    print("RT", audio_length * 1 / start.elapsed_time(end) * 1000)

    speech_timestamps = get_speech_regions(test_audio, silero_vad, sampling_rate=SAMPLING_RATE, return_seconds=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(1):
        speech_timestamps = get_speech_regions(test_audio, silero_vad, sampling_rate=SAMPLING_RATE, return_seconds=True)
    end.record()
    torch.cuda.synchronize()
    # print("speech region timestamps in seconds", speech_timestamps)
    print("RT", audio_length * 1 / start.elapsed_time(end))
