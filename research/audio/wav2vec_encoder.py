from pathlib import Path

import torch
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import get_seqs_and_padding_mask
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model

audio_wav_path, device, dtype = ...
audio_decoder = AudioDecoder(dtype=torch.float32, device=device)
fbank_converter = WaveformToFbankConverter(
    num_mel_bins=80,
    waveform_scale=2**15,
    channel_last=True,
    standardize=True,
    device=device,
    dtype=dtype,
)
collater = Collater(pad_value=1)

model = load_conformer_shaw_model("conformer_shaw", device=device, dtype=dtype)
model.eval()

with Path(audio_wav_path).open("rb") as fb:
    block = MemoryBlock(fb.read())

decoded_audio = audio_decoder(block)
src = collater(fbank_converter(decoded_audio))["fbank"]
seqs, padding_mask = get_seqs_and_padding_mask(src)

with torch.inference_mode():
    seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
    seqs, padding_mask = model.encoder(seqs, padding_mask)
