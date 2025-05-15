import torch
from typing import Optional
import nemo.collections.asr as nemo_asr # Import NeMo
import logging
import warnings
import numpy as np
import soundfile as sf # For writing temporary audio files
import tempfile # For creating temporary files
import os # For file operations like remove

# Configure logging
# Use a more specific logger name if desired, e.g., logging.getLogger("ParakeetASR")
logger = logging.getLogger(__name__) # Using __name__ is a common practice
# Set level for NeMo's logger
logging.getLogger("nemo_toolkit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='pytorch_lightning.*') # More specific warning ignore
warnings.filterwarnings("ignore", category=FutureWarning)


class ParakeetProcessor:
    def __init__(self):
        """Initialize the Parakeet ASR processor."""
        logger.info("Initializing Parakeet ASR processor...")
        self.asr_model = None # Initialize as None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.setup_model()
        
    def setup_model(self):
        """Initialize the Parakeet model."""
        logger.info(f"Setting up Parakeet ASR model on device: {self.device}")
        try:
            model_id = "nvidia/parakeet-tdt-0.6b-v2"
            # NeMo will handle downloading the model if it's not cached locally.
            # Ensure you have an internet connection the first time this runs.
            # The model requires at least 2GB RAM as per its model card.
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
            self.asr_model.to(self.device)
            self.asr_model.eval() # Set the model to evaluation mode

            logger.info(f"Parakeet ASR model ({model_id}) initialized and moved to {self.device}.")

        except Exception as e:
            logger.error(f"Error initializing Parakeet ASR model: {e}", exc_info=True)
            # Depending on desired behavior, you might re-raise or handle this
            # such that the application can continue without ASR or exit gracefully.
            raise # Re-raise to make the calling code aware of the failure

    def _preprocess_audio(self, audio_data_np: np.ndarray, expected_sample_rate: int = 16000) -> np.ndarray:
        """
        Prepares audio data for the Parakeet ASR model.
        Ensures audio is a 1D float32 NumPy array at the expected sample rate.
        The Parakeet model card specifies 16kHz mono channel audio.
        """
        if audio_data_np is None or audio_data_np.size == 0:
            logger.warning("Preprocessing received empty audio data.")
            return np.array([], dtype=np.float32)

        # Ensure it's a NumPy array
        if not isinstance(audio_data_np, np.ndarray):
            logger.warning("Audio data is not a NumPy array. Attempting conversion.")
            try:
                audio_data_np = np.array(audio_data_np)
            except Exception as e:
                logger.error(f"Failed to convert audio data to NumPy array: {e}", exc_info=True)
                return np.array([], dtype=np.float32)
        
        # Ensure it's 1D (mono)
        if audio_data_np.ndim > 1:
            logger.warning(f"Audio data has {audio_data_np.ndim} dimensions. Converting to mono by taking the mean or first channel.")
            # Example: take the mean across channels if stereo, or adapt as needed
            if audio_data_np.shape[0] < audio_data_np.shape[1]: # (channels, samples)
                audio_data_np = np.mean(audio_data_np, axis=0)
            else: # (samples, channels)
                audio_data_np = np.mean(audio_data_np, axis=1)


        # Convert to float32 if not already
        if audio_data_np.dtype != np.float32:
            if np.issubdtype(audio_data_np.dtype, np.integer):
                # Normalize integer types to [-1, 1] before converting to float32
                # Common for int16 from PyAudio
                max_val = np.iinfo(audio_data_np.dtype).max
                audio_data_np = audio_data_np.astype(np.float32) / max_val
            else:
                # For other float types, just convert
                audio_data_np = audio_data_np.astype(np.float32)
        
        # Basic normalization: ensure values are roughly within [-1, 1]
        # This step might be redundant if your input audio is already well-normalized.
        # NeMo models are generally robust, but good practice.
        abs_max = np.abs(audio_data_np).max()
        if abs_max > 1.0:
            logger.debug(f"Audio data max absolute value {abs_max} > 1.0. Normalizing.")
            audio_data_np /= abs_max
        elif abs_max == 0: # Avoid division by zero for pure silence
            logger.debug("Audio data is pure silence.")
            # audio_data_np remains all zeros

        logger.debug(f"Preprocessed audio for Parakeet - Shape: {audio_data_np.shape}, Type: {audio_data_np.dtype}, Range: [{audio_data_np.min():.3f}, {audio_data_np.max():.3f}]")
        return audio_data_np

    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribes a NumPy array of audio data.
        Audio data should be 16kHz mono.
        """
        if self.asr_model is None:
            logger.error("ASR model not initialized. Cannot transcribe.")
            return None
            
        if audio_data is None or audio_data.size == 0:
            logger.info("Received empty audio data for transcription.")
            return None

        # Preprocess audio (ensure it's a NumPy array at 16kHz, float32, mono)
        # Your VoiceCommandSystem provides audio_data as a NumPy array and uses 16kHz.
        audio_processed_np = self._preprocess_audio(audio_data, expected_sample_rate=sample_rate)
        
        if audio_processed_np.size == 0:
            logger.warning("Audio processing resulted in empty data. Skipping transcription.")
            return None
            
        temp_file_path = None # Define here for broader scope in finally block
        try:
            # NeMo's transcribe method primarily takes a list of audio file paths.
            # Saving the processed NumPy array to a temporary WAV file is a robust way.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                sf.write(tmp_audio_file.name, audio_processed_np, sample_rate)
                temp_file_path = tmp_audio_file.name
            
            logger.debug(f"Transcribing temporary audio file: {temp_file_path}")

            # Transcribe using the NeMo model.
            # The `transcribe` method returns a list of transcriptions.
            # For a single audio file, it's a list containing one transcription string.
            # If `return_hypotheses` is True, the structure is more complex.
            # Based on Parakeet model card: `output = asr_model.transcribe(['audio.wav'])`
            # `output[0].text` or `output[0]` (if `return_hypotheses=False` which is default).
            # Let's assume the simpler case for now.
            transcription_results = self.asr_model.transcribe([temp_file_path])
            
            transcribed_text = None
            if transcription_results and isinstance(transcription_results, list) and len(transcription_results) > 0:
                # The result for a single file is typically a list containing one string (the transcription).
                # Or if return_hypotheses=True (default for some models), it's a list of Hypothesis objects.
                # Let's check the type of the first element.
                first_result = transcription_results[0]
                if isinstance(first_result, str):
                    transcribed_text = first_result
                elif hasattr(first_result, 'text'): # Handles Hypothesis object
                    transcribed_text = first_result.text
                else:
                    # If the result structure is different (e.g., nested lists for batched input)
                    # you might need to adjust. For a single file, it's usually simple.
                    # For `parakeet-tdt-0.6b-v2`, `transcribe()` returns List[str] by default.
                    logger.warning(f"Unexpected transcription result format: {type(first_result)}. Full result: {transcription_results}")
                    transcribed_text = str(first_result) # Fallback to string conversion

                # Parakeet includes punctuation and capitalization.
                logger.info(f"Transcribed by Parakeet: '{transcribed_text}'")

            else:
                logger.info("Parakeet transcription returned no result or an empty result.")
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error during Parakeet transcription: {e}", exc_info=True)
            return None
        finally:
            # Clean up the temporary file in all cases (success or error)
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Temporary audio file {temp_file_path} removed.")
                except Exception as cleanup_e:
                    logger.error(f"Error cleaning up temporary audio file {temp_file_path}: {cleanup_e}", exc_info=True)
