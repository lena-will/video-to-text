# Housekeeping ---------------------------------------------------------------------------------------------------------
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Load whisper model and transcripe wav file ---------------------------------------------------------------------------
device = "cpu"
torch_dtype = torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

processor = AutoProcessor.from_pretrained(model_id)

whisper = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

transcription = whisper("test_file.wav")

print(transcription["text"])



