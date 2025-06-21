# core/voice_system.py
import numpy as np
import warnings
import logging
import pyaudio
import queue
import asyncio
import threading
import psutil
import inspect # Not explicitly used in the provided snippet, but good to keep if other parts rely on it
import time
from typing import Optional, Callable, Awaitable, Any, Coroutine

from webrtcvad import Vad
# Import the renamed/updated ASR processor
from speech.whisper_processor import ParakeetProcessor # CORRECTED IMPORT

logger = logging.getLogger(__name__)

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
    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 speak_func: Callable[[str], Awaitable[None]],
                 command_trigger_func: Callable[[str], Coroutine[Any, Any, None]]):
        """
        Initialize the voice command system components.
        Requires the main asyncio event loop, an async speak function,
        and an async function to trigger command processing.
        """
        logger.info("Initializing voice command system...")
        self.loop = loop
        self.speak_func = speak_func
        self.command_trigger_func = command_trigger_func
        self.transcript_callback: Optional[Callable[[str, str], None]] = None
        self.input_device_index: Optional[int] = None
        
        # Use the new processor name
        self.asr_processor: Optional[ParakeetProcessor] = None
        
        self.p: Optional[pyaudio.PyAudio] = None
        self.vad: Optional[Vad] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.background_stream: Optional[pyaudio.Stream] = None
        self.audio_queue: queue.Queue[Any] = queue.Queue() # Not used in current flow, but kept
        self.is_recording: bool = False
        self.is_background_listening: bool = False
        self.current_audio: list[np.ndarray] = []
        self.prev_frames: list[np.ndarray] = [] # For rolling buffer
        self.running: bool = True # General flag for controlling loops/callbacks

        # Audio parameters
        self.sample_rate: int = 16000
        self.channels: int = 1
        self.format: int = pyaudio.paInt16 # paInt16 for 16-bit audio
        self.frame_duration: int = 30 # ms, for VAD
        self.frame_length: int = 0 # samples per VAD frame, calculated in setup_vad
        self.chunk_size: int = 0 # samples per PyAudio read, often same as frame_length
        
        self.prev_frames_duration: float = 0.5 # seconds of audio to keep in rolling buffer
        self.prev_frames_maxlen: int = 0 # max VAD frames in rolling buffer, calculated in setup_vad
        
        # VAD parameters (example values, might need tuning)
        self.silence_limit: float = 0.7 # seconds of silence to stop recording (if implementing VAD-based stop)
        self.min_silence_detections: int = 0 # VAD frames for silence_limit, calculated in setup_vad

        try:
            self.setup_system()
            self.start_background_listener() # Start listening to fill the rolling buffer
        except Exception as e:
            logger.error(f"Critical error during VoiceCommandSystem setup: {e}", exc_info=True)
            self.cleanup() # Attempt to clean up resources
            raise # Re-raise the exception to signal failure to the caller

        logger.info("Voice system initialization complete.")

    def setup_system(self):
        """Set up all system components: PyAudio, VAD, ASR."""
        logger.debug("Setting up system components...")
        self.p = pyaudio.PyAudio()
        self._detect_audio_devices() # Find a suitable input device
        
        # Initialize the ASR processor
        self.asr_processor = ParakeetProcessor() # CORRECT INSTANTIATION
        
        self.setup_vad() # Initialize VAD settings (calculates frame_length, etc.)
        self.chunk_size = self.frame_length # PyAudio will read chunks matching VAD frame size
        logger.debug("System components setup finished.")

    def _detect_audio_devices(self):
        logger.info("Detecting audio input devices...")
        if not self.p: 
            logger.error("PyAudio not initialized before _detect_audio_devices.")
            self.p = pyaudio.PyAudio() # Attempt re-init

        self.input_devices = [] # To store info about all compatible devices
        default_system_device_index: Optional[int] = None
        host_api_info = None
        default_host_api_index = 0 # Default to the first host API

        try:
            # Try to get default host API and its default input device
            default_host_api_info_dict = self.p.get_default_host_api_info()
            default_host_api_index = default_host_api_info_dict['index']
            host_api_info = self.p.get_host_api_info_by_index(default_host_api_index)
            default_system_device_index = host_api_info.get('defaultInputDevice')
            if default_system_device_index == -1: # No default input device for this API
                default_system_device_index = None
            logger.info(f"Default Host API: {host_api_info.get('name')} (Index: {default_host_api_index}), Default System Input Device Index for this API: {default_system_device_index}")
        except Exception as e:
            logger.warning(f"Could not get default device via Host API info: {e}. Will attempt to find default input directly or iterate all devices.")
            try:
                # Fallback: Try to get default input device info directly
                default_input_device_info = self.p.get_default_input_device_info()
                default_system_device_index = default_input_device_info['index']
                default_host_api_index = default_input_device_info['hostApi'] # Use API of this default device
                logger.info(f"Found default input device directly: {default_input_device_info.get('name')} (Index: {default_system_device_index}, Belongs to API Index: {default_host_api_index})")
            except Exception as e2:
                logger.warning(f"Could not get default input device info directly: {e2}. Will iterate all devices and pick first compatible on any API if necessary.")
                default_system_device_index = None # Unset if direct fetch fails

        device_count = self.p.get_device_count()
        logger.debug(f"Total audio devices found by PyAudio: {device_count}")
        selected_device_info = None # To store info of the chosen device

        # Iterate through devices to find a compatible one
        # Prefer devices on the default_host_api_index if determined, otherwise check all
        for i in range(device_count):
            try:
                device_info = self.p.get_device_info_by_index(i)
                # Check if the device belongs to the preferred host API (if one was identified)
                # AND if it has input channels
                is_on_preferred_api = (device_info.get('hostApi') == default_host_api_index)
                has_input_channels = device_info.get('maxInputChannels', 0) >= self.channels

                if has_input_channels: # Must have input channels
                    # If we have a preferred API, only consider devices from it initially
                    # If not, or if we are on a fallback scan, consider any compatible device
                    if (default_host_api_index is not None and is_on_preferred_api) or \
                       (default_host_api_index is None):
                        try:
                            supported = self.p.is_format_supported(
                                rate=self.sample_rate,
                                input_device=device_info['index'],
                                input_channels=self.channels,
                                input_format=self.format
                            )
                            if supported:
                                logger.debug(f"Compatible input device found: {device_info.get('name')} (Index: {i}) on API {device_info.get('hostApi')}")
                                self.input_devices.append(device_info)
                                # Prioritize the system's default input device if it's compatible
                                if default_system_device_index == i:
                                    self.input_device_index = i
                                    selected_device_info = device_info
                                    logger.info(f"Selecting system default input device: {device_info.get('name')} (Index: {i})")
                                    break # Found system default, no need to search further on this API
                                # If no system default yet selected, take the first compatible one found
                                elif self.input_device_index is None:
                                    self.input_device_index = device_info['index']
                                    selected_device_info = device_info
                        except ValueError: # Indicates format not supported by this device
                            logger.debug(f"Format (Rate: {self.sample_rate}, Ch: {self.channels}, Fmt: {self.format}) not supported by device {device_info.get('name')} (Index: {i}). Skipping.")
                        except OSError as pa_os_err: # Other PyAudio errors for this device
                            logger.debug(f"PyAudio OS-level error checking format support for device {i}: {pa_os_err}. Skipping.")
            except Exception as dev_e: # Error getting device info
                logger.warning(f"Could not query full device info for index {i}: {dev_e}")
        
        # If after checking preferred API, no device was selected, try any compatible device
        if self.input_device_index is None and default_host_api_index is not None:
            logger.info(f"No compatible device found on preferred API {default_host_api_index}. Scanning all devices...")
            for dev_info_stored in self.input_devices: # Re-check already found compatible devices from any API
                if self.p.is_format_supported(rate=self.sample_rate, input_device=dev_info_stored['index'], input_channels=self.channels, input_format=self.format):
                    self.input_device_index = dev_info_stored['index']
                    selected_device_info = dev_info_stored
                    logger.info(f"Selected first compatible device from scan: {selected_device_info.get('name')} (Index: {self.input_device_index})")
                    break


        if self.input_device_index is None:
            final_error_msg = f"No compatible audio input devices found supporting {self.sample_rate} Hz, {self.channels} channel(s), 16-bit PCM."
            logger.error(final_error_msg)
            raise RuntimeError(final_error_msg)
        else:
            if not selected_device_info: # Should be set if input_device_index is not None
                try: selected_device_info = self.p.get_device_info_by_index(self.input_device_index)
                except Exception: selected_device_info = {"name": f"Unknown device at Index {self.input_device_index}"} # Fallback
            logger.info(f"Using audio input device: {selected_device_info.get('name', 'N/A')} (Index: {self.input_device_index})")


    def setup_vad(self):
        logger.debug("Setting up Voice Activity Detection (VAD)...")
        if not self.sample_rate: 
            raise ValueError("Sample rate must be set before VAD setup.")
        try:
            self.vad = Vad(3) # Aggressiveness mode 3 (highest)
            if self.frame_duration not in [10, 20, 30]:
                logger.warning(f"VAD frame duration {self.frame_duration}ms is invalid for WebRTCVAD. Setting to 30ms.")
                self.frame_duration = 30
            # Calculate samples per VAD frame
            self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
            # Calculate max VAD frames for the rolling buffer
            self.prev_frames_maxlen = int(self.prev_frames_duration * self.sample_rate / self.frame_length)
            # Calculate VAD frames for silence detection (if using VAD-based stop)
            self.min_silence_detections = int(self.silence_limit * 1000 / self.frame_duration)
            logger.debug(f"VAD setup: Frame Duration={self.frame_duration}ms, Samples/Frame={self.frame_length}, Rolling Buffer={self.prev_frames_maxlen} VAD frames, Silence Thresh={self.min_silence_detections} VAD frames.")
        except Exception as e:
            logger.error(f"Failed to initialize WebRTCVAD: {e}", exc_info=True)
            self.vad = None # Ensure VAD is None if setup fails
            raise # Propagate error


    def set_transcript_callback(self, callback: Callable[[str, str], None]):
        """Set callback for printing transcript/status updates (text, source_type)."""
        self.transcript_callback = callback

    def start_background_listener(self):
        if not self.p: 
            logger.error("PyAudio not initialized. Cannot start background listener.")
            return
        if self.input_device_index is None: 
            logger.error("No input device selected. Cannot start background listener.")
            return
        if self.vad is None:
            logger.error("VAD not initialized. Cannot start background listener effectively (though it will technically run).")
            # Decide if you want to proceed without VAD or return. For rolling buffer, VAD isn't strictly needed.

        if self.background_stream is None and not self.is_background_listening:
            logger.debug("Starting background listener for rolling audio buffer...")
            try:
                self.is_background_listening = True
                # Optional: Wait briefly if espeak is active to avoid initial feedback in buffer
                # if is_espeak_running(): logger.debug("espeak active, waiting briefly..."); time.sleep(0.2)

                self.background_stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.chunk_size, # Use calculated chunk_size
                    stream_callback=self.background_callback
                )
                if self.background_stream.is_active():
                    logger.debug("Background listener stream started successfully.")
                else:
                    logger.error("Background listener stream failed to start (not active after open).")
                    self.is_background_listening = False
                    self.background_stream = None # Ensure it's reset
            except Exception as e:
                logger.error(f"Failed to start background listener stream: {e}", exc_info=True)
                self.background_stream = None
                self.is_background_listening = False
        elif self.is_background_listening:
            logger.debug("Background listener is already running.")
        elif self.background_stream is not None:
            logger.warning("Background stream exists but listener flag is false. Resetting.")
            self.is_background_listening = False # Ensure consistency
            self.stop_background_listener()
            self.start_background_listener() # Try restarting


    def stop_background_listener(self):
        if self.background_stream is not None:
            logger.debug("Stopping background listener...")
            self.is_background_listening = False # Set flag first
            
            stream_to_close = self.background_stream
            self.background_stream = None # Clear reference immediately
            
            try:
                # Check if stream object is valid and has methods before calling
                if hasattr(stream_to_close, 'is_active') and stream_to_close.is_active():
                    if hasattr(stream_to_close, 'stop_stream'): stream_to_close.stop_stream()
                if hasattr(stream_to_close, 'close'): stream_to_close.close()
                logger.debug("Background listener stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error closing background stream: {e}", exc_info=True)
        else:
            logger.debug("Background listener already stopped or not initialized.")
            self.is_background_listening = False # Ensure flag is consistent

    def background_callback(self, in_data, frame_count, time_info, status_flags):
        if not self.is_background_listening: 
            return (None, pyaudio.paComplete) # Signal to stop if flag is false
        
        if status_flags: 
            logger.warning(f"Background audio input status flags non-zero: {status_flags}")

            # Decide how to handle errors, e.g., input overflow. paContinue might still be okay.
            if status_flags == pyaudio.paInputOverflow:
                logger.warning("Input overflow in background callback.")
            # You might want to return paAbort or paComplete on certain errors.
            return (None, pyaudio.paContinue) 

        # Avoid buffering audio if espeak is running to prevent feedback loops in the rolling buffer.
        # This is a simple check; more sophisticated echo cancellation might be needed for robust performance.
        if is_espeak_running(): 
            return (in_data, pyaudio.paContinue) # Consume data but don't add to buffer

        try:
            # Convert raw byte data to NumPy array
            audio_chunk = np.frombuffer(in_data, dtype=np.int16) # Assuming paInt16
            
            # Add a copy of the chunk to the rolling buffer
            self.prev_frames.append(audio_chunk.copy())
            
            # Maintain the rolling buffer size
            if len(self.prev_frames) > self.prev_frames_maxlen:
                 self.prev_frames.pop(0) # Remove the oldest frame
        except Exception as e:
            logger.error(f"Error processing audio in background callback: {e}", exc_info=True)
            # Potentially signal paAbort or paError if critical

        return (in_data, pyaudio.paContinue) # Continue streaming

    def start_quick_record(self):
        logger.info("Attempting to start quick recording...")
        if not self.p: 
            logger.error("PyAudio not initialized. Cannot start quick recording.")
            return False
        if self.input_device_index is None: 
            logger.error("No input device selected. Cannot start quick recording.")
            return False
        if self.is_recording: 
            logger.warning("Quick recording is already active.")
            return False # Or True, if already active is considered success

        self.is_recording = True # Set flag immediately
        self.stop_background_listener() # Pause background listener to free up device/resources

        # Initialize current_audio with a copy of the rolling buffer content
        self.current_audio = list(self.prev_frames) 
        logger.debug(f"Initialized quick recording with {len(self.current_audio)} frames from rolling buffer.")

        # Ensure any previous quick record stream is closed
        if self.stream is not None:
            logger.warning("Quick record stream was unexpectedly open. Closing it before starting new one.")
            try:
                 if self.stream.is_active(): self.stream.stop_stream()
                 self.stream.close()
            except Exception as e: logger.error(f"Error closing pre-existing quick record stream: {e}", exc_info=True)
            self.stream = None

        try:
            self.running = True # Flag for the callback to process data
            # if is_espeak_running(): logger.info("Note: espeak is running at the start of quick record.")

            self.stream = self.p.open(
                format=self.format, 
                channels=self.channels, 
                rate=self.sample_rate,
                input=True, 
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size, # Use calculated chunk_size
                stream_callback=self.quick_record_callback
            )
            if self.stream.is_active():
                 logger.info("Quick recording stream started successfully.")
                 return True
            else:
                 logger.error("Quick recording stream failed to start (not active after open).")
                 self.is_recording = False # Reset flag
                 self.stream = None # Ensure stream is None
                 self.start_background_listener() # Restart background listener if quick record failed
                 return False
        except Exception as e:
            logger.error(f"Failed to start quick recording stream: {e}", exc_info=True)
            self.stream = None
            self.is_recording = False
            self.start_background_listener() # Restart background listener on error
            return False

    def stop_quick_record(self):
        if not self.is_recording:
            logger.debug("Stop quick recording called, but not currently recording.")
            # Ensure background listener is running if it should be
            if not self.is_background_listening: 
                logger.debug("Restarting background listener as it was not active.")
                self.start_background_listener()
            return

        logger.info("Stopping quick recording...")
        self.is_recording = False # Set flag first to stop callback processing new data

        # Stop and close the quick record stream
        if self.stream is not None:
            stream_to_close = self.stream
            self.stream = None # Clear reference
            self.running = False # Also signals callback to complete if it checks this
            try:
                if hasattr(stream_to_close, 'is_active') and stream_to_close.is_active():
                    if hasattr(stream_to_close, 'stop_stream'): stream_to_close.stop_stream()
                if hasattr(stream_to_close, 'close'): stream_to_close.close()
                logger.debug("Quick recording stream stopped and closed.")
            except Exception as e: 
                logger.error(f"Error stopping/closing quick record stream: {e}", exc_info=True)
        
        # Process the recorded audio
        if self.current_audio:
            logger.debug(f"Concatenating {len(self.current_audio)} recorded audio chunks for processing.")
            try:
                # Ensure all elements are indeed NumPy arrays before concatenation
                if all(isinstance(chunk, np.ndarray) for chunk in self.current_audio):
                    combined_audio_data = np.concatenate(self.current_audio)
                    logger.debug(f"Combined audio data shape: {combined_audio_data.shape}, dtype: {combined_audio_data.dtype}")
                    # Schedule the asynchronous transcription and command trigger
                    self._schedule_process_speech(combined_audio_data)
                else:
                    # This case should ideally not happen if callbacks append correctly
                    logger.error("Audio buffer for quick record contains non-NumPy elements. Cannot process.")
                    # For debugging: logger.debug(f"Buffer contents types: {[type(c) for c in self.current_audio]}")
            except ValueError as e: # Handles errors from np.concatenate (e.g. empty list, incompatible shapes if logic changes)
                logger.error(f"Error concatenating audio chunks: {e}. Number of chunks: {len(self.current_audio)}", exc_info=True)
            except Exception as e: # Catch any other unexpected errors during combination
                logger.error(f"Unexpected error during audio data combination for processing: {e}", exc_info=True)
            finally:
                 self.current_audio = [] # Clear buffer whether processing succeeded or failed
        else:
            logger.info("No audio was captured during the quick record session (current_audio buffer is empty).")
            # If no audio, still call transcript callback with a silence indicator if a callback is set
            if self.transcript_callback and self.loop and self.loop.is_running():
                 self.loop.call_soon_threadsafe(self.transcript_callback, "...", "Voice") # Indicate silence or no input

        # Crucially, restart the background listener *after* audio processing is scheduled (or determined not needed)
        self.start_background_listener()
        logger.debug("Background listener restarted after quick record stop.")


    def quick_record_callback(self, in_data, frame_count, time_info, status_flags):
        if not self.running or not self.is_recording: 
            return (None, pyaudio.paComplete) # Stop stream if not supposed to be running/recording
        
        if status_flags: 
            logger.warning(f"Quick record audio input status flags non-zero: {status_flags}")
            if status_flags == pyaudio.paInputOverflow:
                logger.warning("Input overflow in quick record callback.")
            return (None, pyaudio.paContinue) # Decide if to continue or abort on flags

        # Avoid adding audio to buffer if espeak is running (simple feedback prevention)
        if is_espeak_running(): 
            return (in_data, pyaudio.paContinue)

        try:
            audio_chunk = np.frombuffer(in_data, dtype=np.int16) # Assuming paInt16
            self.current_audio.append(audio_chunk.copy()) # Add new chunk to the recording buffer
        except Exception as e:
            logger.error(f"Error in quick record audio callback while processing data: {e}", exc_info=True)

        return (in_data, pyaudio.paContinue) # Continue stream

    def _schedule_process_speech(self, audio_data: np.ndarray):
        """Schedules the async process_speech method in the main event loop."""
        if not self.loop or not self.loop.is_running():
            logger.error("Cannot schedule speech processing: Event loop not available or not running.")
            return
        try:
            # Create a task for the asynchronous speech processing
            # This ensures it runs as part of the asyncio event loop management
            self.loop.call_soon_threadsafe(
                lambda: self.loop.create_task(self.process_speech(audio_data))
            )
            logger.debug("Scheduled process_speech task in event loop.")
        except Exception as e:
            logger.error(f"Error scheduling speech processing task via call_soon_threadsafe: {e}", exc_info=True)

    async def process_speech(self, audio_data: np.ndarray):
        """
        Transcribes speech using the ASR processor and triggers command processing via callback.
        Also calls the transcript_callback for printing.
        """
        logger.debug(f"Starting asynchronous speech processing for audio data of shape: {audio_data.shape}")
        
        # Check necessary components are initialized
        if not self.asr_processor: # CORRECTED CHECK
             logger.error("Cannot process speech: ASR Processor (ParakeetProcessor) is not initialized.")
             if self.transcript_callback: self.transcript_callback("[Error: ASR system not ready]", "Error")
             return
        if not self.command_trigger_func:
             logger.error("Cannot process speech: Command trigger function is not set.")
             if self.transcript_callback: self.transcript_callback("[Error: Command trigger not set]", "Error")
             return


        transcribed_text: Optional[str] = None
        try:
            # 1. Transcribe the audio data
            # The transcribe method of ParakeetProcessor is expected to be async
            logger.debug("Calling ASR processor's transcribe method...")
            transcribed_text = await self.asr_processor.transcribe(audio_data) # CORRECTED USAGE
            logger.debug(f"ASR transcription result: '{transcribed_text}'")


            # 2. Print Transcript (using the dedicated transcript_callback)
            if self.transcript_callback:
                text_to_print = transcribed_text if transcribed_text else "..." # Use "..." if no text
                # Schedule this to be printed on the main thread via the print queue
                self.transcript_callback(text_to_print, "Voice") 

            # 3. Trigger Command Processing (if transcription was successful and yielded text)
            if transcribed_text:
                logger.info(f"Transcription successful: '{transcribed_text}'. Triggering command processing.")
                # Call the async command_trigger_func (passed during __init__)
                await self.command_trigger_func(transcribed_text)
            else:
                logger.info("Transcription returned no text or failed. Skipping command trigger.")

        except asyncio.CancelledError:
             logger.info("Speech processing task was cancelled.")
             # Optionally print cancellation message via transcript callback
             if self.transcript_callback: self.transcript_callback("[Speech processing cancelled by system]", "System")
             # Do not trigger command processing if task was cancelled
        except Exception as e:
            logger.error(f"Error during asynchronous speech processing pipeline: {e}", exc_info=True)
            # Use transcript callback safely to report the error to the user
            if self.transcript_callback:
                error_msg_for_user = f"[Error processing speech: {type(e).__name__}]"
                # Avoid printing potentially long/complex exception details directly to user.
                # Logged details are available for debugging.
                if transcribed_text is not None: # If transcription happened before error
                    error_msg_for_user += f" (Partial/Attempted Text: '{transcribed_text[:50]}...')"
                self.transcript_callback(error_msg_for_user, "Error")
            # Do not trigger command processing if an error occurred

    def cleanup(self):
        logger.info("Cleaning up VoiceCommandSystem resources...")
        self.running = False # Signal all loops/callbacks to stop
        self.is_background_listening = False
        self.is_recording = False

        # Close PyAudio streams safely
        # Stop background listener first as it might be more active
        self.stop_background_listener() 

        # Then ensure the main recording stream is also stopped and closed
        if self.stream is not None:
            logger.debug("Cleaning up main recording stream...")
            stream_to_close = self.stream
            self.stream = None
            try:
                if hasattr(stream_to_close, 'is_active') and stream_to_close.is_active():
                    if hasattr(stream_to_close, 'stop_stream'): stream_to_close.stop_stream()
                if hasattr(stream_to_close, 'close'): stream_to_close.close()
                logger.debug("Main recording stream closed.")
            except Exception as e:
                logger.error(f"Exception closing main recording stream during cleanup: {e}", exc_info=True)
        
        # Terminate PyAudio instance
        if self.p:
            logger.debug("Terminating PyAudio instance...")
            try:
                self.p.terminate()
                logger.debug("PyAudio instance terminated.")
            except Exception as e:
                logger.error(f"Error terminating PyAudio instance: {e}", exc_info=True)
            finally:
                self.p = None # Ensure it's None after attempting termination

        # Any other cleanup, e.g., clearing queues or buffers if they hold large data
        self.current_audio = []
        self.prev_frames = []
        if self.audio_queue: # if you were using it
            while not self.audio_queue.empty():
                try: self.audio_queue.get_nowait()
                except queue.Empty: break
            logger.debug("Audio queue cleared.")

        logger.info("Voice command system resources released.")
