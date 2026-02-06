import base64
import logging
import os
import warnings

import torch
import torchaudio

# Suppress specific warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# Import the consolidated model definitions
import fun_asr_gguf.model_definition as model_def

# =========================================================================
# Configuration
# =========================================================================

OUTPUT_DIR = r"./model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_dir = r"./Fun-ASR-Nano-2512"
weight_path = os.path.join(model_dir, "model.pt")

# New standardized names
onnx_encoder_fp32 = f"{OUTPUT_DIR}/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx"
onnx_ctc_fp32 = f"{OUTPUT_DIR}/Fun-ASR-Nano-CTC.fp32.onnx"
tokens_path = f"{OUTPUT_DIR}/tokens.txt"

SAMPLE_RATE = 16000
NFFT_STFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
OPSET = 18

# =========================================================================
# Utils
# =========================================================================


def generate_sensevoice_vocab(tiktoken_path):
    print(f"Generating vocabulary from {tiktoken_path}...")
    tokens = []
    with open(tiktoken_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tokens.append(line.split()[0])

    special_labels = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        "en",
        "zh",
        "de",
        "es",
        "ru",
        "ko",
        "fr",
        "ja",
        "pt",
        "tr",
        "pl",
        "ca",
        "nl",
        "ar",
        "sv",
        "it",
        "id",
        "hi",
        "fi",
        "vi",
        "he",
        "uk",
        "el",
        "ms",
        "cs",
        "ro",
        "da",
        "hu",
        "ta",
        "no",
        "th",
        "ur",
        "hr",
        "bg",
        "lt",
        "la",
        "mi",
        "ml",
        "cy",
        "sk",
        "te",
        "fa",
        "lv",
        "bn",
        "sr",
        "az",
        "sl",
        "kn",
        "et",
        "mk",
        "br",
        "eu",
        "is",
        "hy",
        "ne",
        "mn",
        "bs",
        "kk",
        "sq",
        "sw",
        "gl",
        "mr",
        "pa",
        "si",
        "km",
        "sn",
        "yo",
        "so",
        "af",
        "oc",
        "ka",
        "be",
        "tg",
        "sd",
        "gu",
        "am",
        "yi",
        "lo",
        "uz",
        "fo",
        "ht",
        "ps",
        "tk",
        "nn",
        "mt",
        "sa",
        "lb",
        "my",
        "bo",
        "tl",
        "mg",
        "as",
        "tt",
        "haw",
        "ln",
        "ha",
        "ba",
        "jw",
        "su",
        "yue",
        "minnan",
        "wuyu",
        "dialect",
        "zh/en",
        "en/zh",
        "ASR",
        "AED",
        "SER",
        "Speech",
        "/Speech",
        "BGM",
        "/BGM",
        "Laughter",
        "/Laughter",
        "Applause",
        "/Applause",
        "HAPPY",
        "SAD",
        "ANGRY",
        "NEUTRAL",
        "translate",
        "transcribe",
        "startoflm",
        "startofprev",
        "nospeech",
        "notimestamps",
    ]
    for label in special_labels:
        if not label.startswith("<|"):
            label = f"<|{label}|>"
        tokens.append(base64.b64encode(label.encode()).decode())
    for i in range(1, 51):
        tokens.append(base64.b64encode(f"<|SPECIAL_TOKEN_{i}|>".encode()).decode())
    for i in range(1500):
        tokens.append(base64.b64encode(f"<|{i * 0.02:.2f}|>".encode()).decode())
    tokens.append(base64.b64encode("<blk>".encode()).decode())
    return tokens


def pad_audio_to_min_length(audio_tensor, min_duration=1.0):
    """
    对音频张量进行 padding，使其至少达到指定的最小持续时间。

    Args:
        audio_tensor (torch.Tensor): 输入音频张量，形状为 [batch, channels, samples]。
        min_duration (float): 最小持续时间（单位：秒）。

    Returns:
        torch.Tensor: Padding 后的音频张量。
    """
    min_samples = int(SAMPLE_RATE * min_duration)
    current_samples = audio_tensor.shape[-1]

    if current_samples < min_samples:
        # 计算需要填充的样本数
        pad_length = min_samples - current_samples
        # 使用零值填充
        padding = torch.zeros(*audio_tensor.shape[:-1], pad_length)
        padded_audio = torch.cat([audio_tensor, padding], dim=-1)
        return padded_audio
    else:
        return audio_tensor


# =========================================================================
# Main Export
# =========================================================================


def main():
    print("\n[Hybrid Export] Consolidated & Paddable Model Export...")

    tiktoken_path = os.path.join(model_dir, "multilingual.tiktoken")
    if os.path.exists(tiktoken_path):
        tokens = generate_sensevoice_vocab(tiktoken_path)
        with open(tokens_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(tokens):
                f.write(f"{t} {i}\n")
    else:
        print("Warning: tiktoken file not found, vocab generation skipped.")
        tokens = ["dummy"] * 60515

    hybrid = model_def.HybridSenseVoice(vocab_size=len(tokens))
    hybrid.load_weights(weight_path)
    hybrid.eval()

    stft = model_def.STFT_Process(
        n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH
    ).eval()
    fbank = (
        (
            torchaudio.functional.melscale_fbanks(
                NFFT_STFT // 2 + 1,
                20,
                SAMPLE_RATE // 2,
                N_MELS,
                SAMPLE_RATE,
                None,
                "htk",
            )
        )
        .transpose(0, 1)
        .unsqueeze(0)
    )

    with torch.no_grad():
        print("\n[1/2] Exporting Paddable Encoder-Adaptor (Dynamo=True)...")
        # Use the upgraded Paddable Wrapper
        enc_wrapper = model_def.EncoderExportWrapperPaddable(hybrid, stft, fbank).eval()

        # 定义 dummy 输入（极短音频）
        dummy_samples = int(SAMPLE_RATE * 0.5)  # 修复：显式转为整数
        audio = torch.randn(1, 1, dummy_samples)
        ilens = torch.tensor([dummy_samples], dtype=torch.long)

        # 对音频进行 padding
        audio_padded = pad_audio_to_min_length(audio, min_duration=1.0)
        ilens_padded = torch.tensor([audio_padded.shape[-1]], dtype=torch.long)

        torch.onnx.export(
            enc_wrapper,
            (audio_padded, ilens_padded),
            onnx_encoder_fp32,
            input_names=["audio", "ilens"],
            output_names=["enc_output", "adaptor_output"],
            dynamic_axes={
                "audio": {2: "samples"},
                "ilens": {0: "batch"},
                "enc_output": {1: "enc_frames"},
                "adaptor_output": {1: "adaptor_frames"},
            },
            opset_version=OPSET,
            dynamo=True,
        )

        print("\n[2/2] Exporting CTC Head (Dynamo=True)...")
        ctc_wrapper = model_def.CTCHeadExportWrapper(hybrid).eval()
        dummy_enc = torch.randn(1, 100, 512)
        torch.onnx.export(
            ctc_wrapper,
            (dummy_enc,),
            onnx_ctc_fp32,
            input_names=["enc_output"],
            output_names=["indices"],
            dynamic_axes={"enc_output": {1: "enc_len"}, "indices": {1: "enc_len"}},
            opset_version=OPSET,
            dynamo=True,
        )

    print("\n[Success] Export complete using consolidated module.")


if __name__ == "__main__":
    main()
