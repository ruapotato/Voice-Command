# core/voice_system.py
import numpy as np
import warnings
import logging
import pyaudio
import queue
import asyncio
import threading
import psutil
import inspect
import time
from typing import Optional, Callable, Awaitable, Any, Coroutine # Added Coroutine

from webrtcvad import Vad
# Import CommandProcessor and WhisperProcessor from their respective locations
# Remove CommandProcessor import if it's no longer used directly here
# from commands.command_processor import CommandProcessor
from speech.whisper_processor import WhisperProcessor

logger = logging.getLogger(__name__)

# is_espeak_running function remains the same...
def is_espeak_running():
    """Check if espeak is currently running"""
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if proc.info['name'] == 'espeak' or \
               (proc.info['cmdline'] and 'espeak' in proc.info['cmdline'][0]):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


class VoiceCommandSystem:
    # <<< MODIFIED __init__ signature >>>
    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 speak_func: Callable[[str], Awaitable[None]],
                 command_trigger_func: Callable[[str], Coroutine[Any, Any, None]]): # New async callback
        """
        Initialize the voice command system components.
        Requires the main asyncio event loop, an async speak function,
        and an async function to trigger command processing.
        """
        logger.info("Initializing voice command system...")
        self.loop = loop
        self.speak_func = speak_func
        self.command_trigger_func = command_trigger_func # <<< STORED command trigger
        self.transcript_callback: Optional[Callable[[str, str], None]] = None # For printing transcripts only now
        self.input_device_index: Optional[int] = None
        # self.command_processor: Optional[CommandProcessor] = None # No longer needed here
        self.whisper: Optional[WhisperProcessor] = None
        self.p: Optional[pyaudio.PyAudio] = None
        self.vad: Optional[Vad] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.background_stream: Optional[pyaudio.Stream] = None
        self.audio_queue: queue.Queue[Any] = queue.Queue()
        self.is_recording: bool = False
        self.is_background_listening: bool = False
        self.current_audio: list[np.ndarray] = []
        self.prev_frames: list[np.ndarray] = []
        self.running: bool = True

        # Audio parameters
        self.sample_rate: int = 16000
        self.channels: int = 1
        self.format: int = pyaudio.paInt16
        self.frame_duration: int = 30
        self.frame_length: int = 0
        self.chunk_size: int = 0
        self.prev_frames_duration: float = 0.5
        self.prev_frames_maxlen: int = 0
        self.silence_limit: float = 0.7
        self.min_silence_detections: int = 0

        try:
            self.setup_system()
            self.start_background_listener()
        except Exception as e:
            logger.error(f"Error during system setup: {e}", exc_info=True)
            self.cleanup()
            raise

        logger.info("Voice system initialization complete.")

    def setup_system(self):
        """Set up all system components."""
        self.p = pyaudio.PyAudio()
        self._detect_audio_devices()
        self.whisper = WhisperProcessor() # Initialize Whisper
        self.setup_vad() # Initialize VAD
        # self.command_processor = CommandProcessor() # Remove: Not initialized here anymore
        self.chunk_size = self.frame_length

    # _detect_audio_devices (no changes needed)
    def _detect_audio_devices(self):
        # ... (keep existing implementation) ...
        logger.info("Detecting audio input devices...")
        if not self.p: self.p = pyaudio.PyAudio()

        self.input_devices = []
        default_system_device_index: Optional[int] = None
        host_api_info = None
        default_host_api_index = 0

        try:
            default_host_api_index = self.p.get_default_host_api_info()['index']
            host_api_info = self.p.get_host_api_info_by_index(default_host_api_index)
            default_system_device_index = host_api_info.get('defaultInputDevice')
            if default_system_device_index == -1: default_system_device_index = None
            logger.info(f"Default Host API: {host_api_info.get('name')}, Default Input Device Index: {default_system_device_index}")
        except Exception as e:
            logger.warning(f"Could not get default device via Host API info: {e}. Will check all devices.")
            try:
                default_info = self.p.get_default_input_device_info()
                default_system_device_index = default_info['index']
                default_host_api_index = default_info['hostApi']
                logger.info(f"Found default input device directly: {default_info.get('name')} (Index: {default_system_device_index}, API: {default_host_api_index})")
            except Exception as e2:
                logger.warning(f"Could not get default input device info directly: {e2}. Falling back to first compatible.")

        device_count = self.p.get_device_count()
        logger.debug(f"Total audio devices found: {device_count}")
        selected_device_info = None

        for i in range(device_count):
            try:
                device_info = self.p.get_device_info_by_index(i)
                if device_info.get('hostApi') == default_host_api_index and \
                   device_info.get('maxInputChannels', 0) >= self.channels:
                    try:
                        supported = self.p.is_format_supported(
                            rate=self.sample_rate,
                            input_device=device_info['index'],
                            input_channels=self.channels,
                            input_format=self.format
                        )
                        if supported:
                            logger.debug(f"Compatible input device found: {device_info.get('name')} (Index: {i}) on API {default_host_api_index}")
                            self.input_devices.append(device_info)
                            if self.input_device_index is None:
                                self.input_device_index = device_info['index']
                                selected_device_info = device_info
                            if default_system_device_index == i:
                                self.input_device_index = i
                                selected_device_info = device_info
                                logger.info(f"Selecting system default input device: {device_info.get('name')} (Index: {i})")
                    except ValueError:
                        logger.debug(f"Could not check format support for device {device_info.get('name')} (Index: {i}). Skipping.")
                    except OSError as pa_os_err:
                        logger.debug(f"Format check failed for device {i} (OSError): {pa_os_err}. Skipping.")
            except Exception as dev_e:
                logger.warning(f"Could not query full device info for index {i}: {dev_e}")

        if self.input_device_index is None:
            logger.error(f"No compatible audio input devices found for Host API {default_host_api_index} supporting {self.sample_rate} Hz!")
            raise RuntimeError("No compatible audio input devices found.")
        else:
            if not selected_device_info:
                try: selected_device_info = self.p.get_device_info_by_index(self.input_device_index)
                except Exception: selected_device_info = {"name": f"Index {self.input_device_index}"}
            logger.info(f"Using input device: {selected_device_info.get('name', 'Unknown')} (Index: {self.input_device_index})")


    # setup_vad (no changes needed)
    def setup_vad(self):
        # ... (keep existing implementation) ...
        logger.debug("Setting up VAD...")
        if not self.sample_rate: raise ValueError("Sample rate must be set before VAD setup.")
        self.vad = Vad(3)
        if self.frame_duration not in [10, 20, 30]:
            logger.warning(f"Invalid VAD frame duration {self.frame_duration}ms. Setting to 30ms.")
            self.frame_duration = 30
        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        self.prev_frames_maxlen = int(self.prev_frames_duration * self.sample_rate / self.frame_length)
        self.min_silence_detections = int(self.silence_limit * 1000 / self.frame_duration)
        logger.debug(f"VAD setup: Frame Length={self.frame_length}, Buffer={self.prev_frames_maxlen} frames, Silence Chunks={self.min_silence_detections}")


    # set_transcript_callback (no changes needed, purpose clarified)
    def set_transcript_callback(self, callback: Callable[[str, str], None]):
        """Set callback ONLY for printing transcript/status updates (text, source_type)."""
        self.transcript_callback = callback

    # start_background_listener (no changes needed)
    def start_background_listener(self):
        # ... (keep existing implementation) ...
         if not self.p: logger.error("PyAudio not initialized in background listener."); return
         if self.input_device_index is None: logger.error("No input device selected for background listener."); return

         if self.background_stream is None and not self.is_background_listening:
             logger.debug("Starting background listener for rolling buffer")
             try:
                 self.is_background_listening = True
                 # if is_espeak_running(): logger.warning("Waiting for espeak..."); time.sleep(0.5) # Optional wait

                 self.background_stream = self.p.open(
                     format=self.format,
                     channels=self.channels,
                     rate=self.sample_rate,
                     input=True,
                     input_device_index=self.input_device_index,
                     frames_per_buffer=self.frame_length,
                     stream_callback=self.background_callback
                 )
                 if self.background_stream.is_active():
                     logger.debug("Background listener stream started successfully.")
                 else:
                     logger.error("Background listener stream failed to start (not active).")
                     self.is_background_listening = False
                     self.background_stream = None
             except Exception as e:
                 logger.error(f"Failed to start background listener: {e}", exc_info=True)
                 self.background_stream = None
                 self.is_background_listening = False

    # stop_background_listener (no changes needed)
    def stop_background_listener(self):
        # ... (keep existing implementation) ...
        if self.background_stream is not None:
            logger.debug("Stopping background listener")
            self.is_background_listening = False
            stream_to_close = self.background_stream
            self.background_stream = None
            try:
                if stream_to_close.is_active(): stream_to_close.stop_stream()
                stream_to_close.close()
                logger.debug("Background listener stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error closing background stream: {e}")

    # background_callback (no changes needed)
    def background_callback(self, in_data, frame_count, time_info, status):
        # ... (keep existing implementation) ...
        if not self.is_background_listening: return (None, pyaudio.paComplete)
        if status: logger.warning(f"Background audio input status non-zero: {status}"); return (None, pyaudio.paContinue)
        # Skip buffering if espeak is running to avoid feedback loops in buffer
        if is_espeak_running(): return (in_data, pyaudio.paContinue)

        try:
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            # Use deque for potentially better performance? Or list is fine.
            self.prev_frames.append(audio_chunk.copy())
            # Efficiently trim the list
            if len(self.prev_frames) > self.prev_frames_maxlen:
                 self.prev_frames.pop(0) # Remove oldest frame
            # Or using slicing (might create new list):
            # self.prev_frames = self.prev_frames[-self.prev_frames_maxlen:]
        except Exception as e:
            logger.error(f"Error in background audio processing: {e}", exc_info=True)

        return (in_data, pyaudio.paContinue)


    # start_quick_record (no changes needed)
    def start_quick_record(self):
        # ... (keep existing implementation) ...
        logger.info("Starting quick recording")
        if not self.p: logger.error("PyAudio not initialized."); return False
        if self.input_device_index is None: logger.error("No input device selected."); return False
        if self.is_recording: logger.warning("Already recording."); return False

        self.is_recording = True
        self.stop_background_listener() # Stop background before starting main record
        self.current_audio = list(self.prev_frames) # Create a copy of the buffer
        logger.debug(f"Added {len(self.current_audio)} frames from rolling buffer")

        if self.stream is not None:
            logger.warning("Quick record stream was already open? Closing it.")
            try:
                 if self.stream.is_active(): self.stream.stop_stream()
                 self.stream.close()
            except Exception as e: logger.error(f"Error closing pre-existing quick record stream: {e}")
            self.stream = None

        try:
            self.running = True
            if is_espeak_running(): logger.info("Espeak running during record start.") # Just log
            self.stream = self.p.open(
                format=self.format, channels=self.channels, rate=self.sample_rate,
                input=True, input_device_index=self.input_device_index,
                frames_per_buffer=self.frame_length, stream_callback=self.quick_record_callback
            )
            if self.stream.is_active():
                 logger.debug("Quick record stream started successfully.")
                 return True
            else:
                 logger.error("Quick record stream failed to start (not active).")
                 self.is_recording = False
                 self.stream = None
                 self.start_background_listener() # Restart background if quick record failed
                 return False
        except Exception as e:
            logger.error(f"Failed to start quick record stream: {e}", exc_info=True)
            self.stream = None
            self.is_recording = False
            self.start_background_listener() # Restart background listener on error
            return False


    # stop_quick_record (no changes needed)
    def stop_quick_record(self):
        # ... (keep existing implementation) ...
        if not self.is_recording:
            logger.debug("Stop quick called but not recording.")
            # Ensure background listener restarts if it's not running
            if not self.is_background_listening: self.start_background_listener()
            return

        logger.info("Stopping quick recording")
        self.is_recording = False # Set flag first

        # Stop and close the quick record stream
        if self.stream is not None:
            stream_to_close = self.stream
            self.stream = None # Clear reference
            self.running = False # Stop callback processing flag
            try:
                if stream_to_close.is_active(): stream_to_close.stop_stream()
                stream_to_close.close()
                logger.debug("Quick record stream stopped and closed.")
            except Exception as e: logger.error(f"Error stopping/closing quick record stream: {e}")

        # Process the recorded audio
        if self.current_audio:
            logger.debug(f"Processing recorded audio of size: {len(self.current_audio)} chunks")
            try:
                # Ensure all elements are numpy arrays before concatenating
                if all(isinstance(chunk, np.ndarray) for chunk in self.current_audio):
                    combined_audio = np.concatenate(self.current_audio)
                    # Schedule the async transcription and command trigger
                    self._schedule_process_speech(combined_audio)
                else:
                    logger.error("Audio buffer corrupted: contains non-numpy elements.")
                    # Optionally print the types for debugging
                    # logger.debug(f"Buffer contents types: {[type(c) for c in self.current_audio]}")
            except ValueError as e:
                # Handle cases like empty list or incompatible shapes if logic changes
                logger.error(f"Error concatenating audio chunks: {e}. Chunks: {len(self.current_audio)}")
            except Exception as e:
                logger.error(f"Unexpected error during audio combination: {e}")
            finally:
                 # Clear buffer regardless of success/failure
                 self.current_audio = []
        else:
            logger.info("No audio recorded during quick record session.")
             # If no audio, still call transcript callback with silence indicator
            if self.transcript_callback:
                 # Schedule the print callback on the main loop
                 self.loop.call_soon_threadsafe(self.transcript_callback, "...", "Voice")


        # Restart background listener *after* potentially scheduling processing
        self.start_background_listener()


    # quick_record_callback (no changes needed)
    def quick_record_callback(self, in_data, frame_count, time_info, status):
        # ... (keep existing implementation) ...
        if not self.running or not self.is_recording: return (None, pyaudio.paComplete)
        if status: logger.warning(f"Quick record audio input status non-zero: {status}"); return (None, pyaudio.paContinue)
        # Skip adding audio if espeak is running (avoids feedback in recording)
        if is_espeak_running(): return (in_data, pyaudio.paContinue)

        try:
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            self.current_audio.append(audio_chunk.copy()) # Add chunk to buffer
        except Exception as e:
            logger.error(f"Error in quick record audio callback: {e}", exc_info=True)

        return (in_data, pyaudio.paContinue)


    def _schedule_process_speech(self, audio_data):
        """Schedules the async process_speech method in the main event loop."""
        if not self.loop or not self.loop.is_running():
            logger.error("Cannot schedule speech processing: Event loop not available/running.")
            return
        try:
            # Create task directly using loop.create_task from the thread-safe call
            self.loop.call_soon_threadsafe(
                lambda: self.loop.create_task(self.process_speech(audio_data))
            )
        except Exception as e:
            logger.error(f"Error scheduling speech processing task: {e}", exc_info=True)

    # --- process_speech MODIFIED ---
    async def process_speech(self, audio_data):
        """
        Transcribes speech and triggers command processing via callback.
        Prints transcript using transcript_callback.
        """
        logger.debug("Processing speech data asynchronously...")
        # Check necessary components
        if not self.whisper or not self.command_trigger_func:
             logger.error("Cannot process speech: Missing Whisper or Command Trigger function.")
             # Still try to print an error via transcript callback if available
             if self.transcript_callback: self.transcript_callback("[Error: Missing components for speech processing]", "Error")
             return

        transcribed_text = None
        try:
            # 1. Transcribe
            transcribed_text = await self.whisper.transcribe(audio_data)

            # 2. Print Transcript (using the dedicated callback)
            if self.transcript_callback:
                text_to_print = transcribed_text if transcribed_text else "..."
                self.transcript_callback(text_to_print, "Voice") # Always print something

            # 3. Trigger Command Processing (if transcription successful)
            if transcribed_text:
                logger.info(f"Transcription successful: '{transcribed_text}'. Triggering command processing.")
                # Call the async function passed during init
                await self.command_trigger_func(transcribed_text)
            else:
                logger.info("Transcription returned no text. Skipping command trigger.")

        except asyncio.CancelledError:
             logger.info("Speech processing task cancelled.")
             # Optionally print cancellation message via transcript callback
             if self.transcript_callback: self.transcript_callback("[Speech processing cancelled]", "System")
             # Do not trigger command if cancelled during transcription
        except Exception as e:
            logger.error(f"Error during async speech processing: {e}", exc_info=True)
            # Use transcript callback safely to report the error
            if self.transcript_callback:
                error_msg = f"[Error processing speech: {e}]"
                # Include transcription attempt if available
                if transcribed_text is not None: error_msg += f" (Text: '{transcribed_text}')"
                self.transcript_callback(error_msg, "Error")
            # Do not trigger command processing on error

    # --- cleanup (no changes needed) ---
    def cleanup(self):
        # ... (keep existing implementation) ...
        logger.info("Cleaning up VoiceCommandSystem resources...")
        self.running = False
        self.is_background_listening = False
        self.is_recording = False

        # Close streams safely
        for stream_ref in [self.stream, self.background_stream]:
            stream = stream_ref
            if stream:
                try:
                    if hasattr(stream, 'is_active') and stream.is_active():
                        if hasattr(stream, 'stop_stream'): stream.stop_stream()
                    if hasattr(stream, 'close'): stream.close()
                except Exception as e:
                    logger.debug(f"Exception closing stream: {e}")
        self.stream = None
        self.background_stream = None

        # Terminate PyAudio
        if self.p:
            try:
                logger.debug("Terminating PyAudio...")
                self.p.terminate()
                logger.debug("PyAudio terminated.")
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self.p = None

        logger.info("Voice command system resources released.")
