"""Whisper-based speech recognition processor."""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging
import warnings
import numpy as np

# Configure logging
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

class WhisperProcessor:
    def __init__(self):
        """Initialize the Whisper processor."""
        print("Initializing Whisper processor...")
        self.setup_model()
        
    def setup_model(self):
        """Initialize the Whisper model and pipeline."""
        try:
            # Setup device
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            print(f"Using device: {self.device}")

            # Load model - using v3 for improved accuracy
            model_id = "openai/whisper-large-v3"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device)

            # Load processor
            processor = AutoProcessor.from_pretrained(model_id)

            # Setup pipeline with optimized parameters
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30,
                stride_length_s=5,
                batch_size=1
            )
            print("Whisper model initialized")

        except Exception as e:
            print(f"Error initializing Whisper: {e}")
            raise

    def _preprocess_audio(self, audio_data):
        """Preprocess audio data for optimal Whisper processing."""
        try:
            # Debug original audio
            print(f"Input audio - Shape: {audio_data.shape}, Type: {audio_data.dtype}, Range: [{audio_data.min()}, {audio_data.max()}]")
            
            # Ensure data is in float32
            audio_float = audio_data.astype(np.float32)
            
            # Apply pre-emphasis filter to enhance high frequencies
            pre_emphasis = 0.97
            emphasized_audio = np.append(
                audio_float[0], 
                audio_float[1:] - pre_emphasis * audio_float[:-1]
            )
            
            # Normalize using RMS normalization
            rms = np.sqrt(np.mean(np.square(emphasized_audio)))
            if rms > 0:
                normalized_audio = emphasized_audio / rms
            else:
                normalized_audio = emphasized_audio
            
            # Clip to prevent extreme values
            normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
            
            # Debug processed audio
            print(f"Processed audio - Shape: {normalized_audio.shape}, Range: [{normalized_audio.min():.3f}, {normalized_audio.max():.3f}]")
            
            return normalized_audio
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None

    async def transcribe(self, audio_data):
        """Process audio data and return transcribed text."""
        try:
            if audio_data is None:
                print("Received empty audio data")
                return None

            # Preprocess audio
            audio_processed = self._preprocess_audio(audio_data)
            if audio_processed is None:
                return None

            # Process with adjusted parameters for better command recognition
            inputs = {
                "raw": audio_processed,
                "sampling_rate": 16000
            }

            print("Processing audio segment...")
            result = self.pipe(
                inputs,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "en",
                    "temperature": 0,  # Deterministic decoding for commands
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6
                }
            )
            
            transcribed_text = result["text"].strip()
            print(f"Transcribed: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
