# core/voice_system.py
import numpy as np
# import torch # Don't import torch here if only WhisperProcessor uses it
import warnings
import logging
import pyaudio
import queue
import asyncio
import threading
import psutil
import inspect # Needed for checks
import time # For sleep
from typing import Optional, Callable, Awaitable, Any # Added Callable, Awaitable, Any

from webrtcvad import Vad
# Import CommandProcessor and WhisperProcessor from their respective locations
from commands.command_processor import CommandProcessor
from speech.whisper_processor import WhisperProcessor

logger = logging.getLogger(__name__)

# is_espeak_running function remains the same...
def is_espeak_running():
    """Check if espeak is currently running"""
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            # Check process name and command line for espeak
            if proc.info['name'] == 'espeak' or \
               (proc.info['cmdline'] and proc.info['cmdline'] and 'espeak' in proc.info['cmdline'][0]):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass # Ignore processes we can't access or that are gone
    return False


class VoiceCommandSystem:
    # <<< ADDED speak_func parameter and type hint >>>
    def __init__(self, loop: asyncio.AbstractEventLoop, speak_func: Callable[[str], Awaitable[None]]):
        """
        Initialize the voice command system components.
        Requires the main asyncio event loop and an async speak function.
        """
        logger.info("Initializing voice command system...")
        self.loop = loop # <<< STORED loop reference
        self.speak_func = speak_func # <<< STORED speak function
        self.transcript_callback: Optional[Callable[[str, str], None]] = None # Callback for printing transcripts/results
        self.input_device_index: Optional[int] = None
        self.command_processor: Optional[CommandProcessor] = None
        self.whisper: Optional[WhisperProcessor] = None
        self.p: Optional[pyaudio.PyAudio] = None
        self.vad: Optional[Vad] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.background_stream: Optional[pyaudio.Stream] = None
        self.audio_queue: queue.Queue[Any] = queue.Queue() # Type hint if needed
        self.is_recording: bool = False
        self.is_background_listening: bool = False
        self.current_audio: list[np.ndarray] = [] # List to hold audio chunks
        self.prev_frames: list[np.ndarray] = [] # Rolling buffer
        self.running: bool = True # Controls audio stream callbacks

        # Audio parameters (can be moved to config later)
        self.sample_rate: int = 16000
        self.channels: int = 1
        self.format: int = pyaudio.paInt16 # Corresponds to np.int16
        self.frame_duration: int = 30 # VAD frame duration in ms
        self.frame_length: int = 0 # Calculated in setup_vad
        self.chunk_size: int = 0 # Usually same as frame_length
        self.prev_frames_duration: float = 0.5 # seconds
        self.prev_frames_maxlen: int = 0 # Calculated in setup_vad
        self.silence_limit: float = 0.7 # seconds
        self.min_silence_detections: int = 0 # Calculated in setup_vad


        # --- Setup Call ---
        try:
            self.setup_system()
            # Start background listener only *after* successful setup
            self.start_background_listener()
        except Exception as e:
             logger.error(f"Error during system setup: {e}", exc_info=True)
             # Cleanup partially initialized resources?
             self.cleanup() # Attempt cleanup if setup fails
             raise # Let main.py know initialization failed

        logger.info("Voice system initialization complete.")

    def setup_system(self):
        """Set up all system components."""
        # Moved initialization from __init__ here
        self.p = pyaudio.PyAudio()
        self._detect_audio_devices() # Sets self.input_device_index

        # Initialize Whisper (can take time)
        self.whisper = WhisperProcessor()

        # Initialize VAD (needs sample_rate)
        self.setup_vad() # Sets self.frame_length, etc.

        # Initialize command processor (no longer needs loop or window)
        self.command_processor = CommandProcessor()

        # Set chunk size based on VAD frame length for audio streams
        self.chunk_size = self.frame_length


    def _detect_audio_devices(self):
        """Detect available audio input devices and select default"""
        logger.info("Detecting audio input devices...")
        if not self.p: self.p = pyaudio.PyAudio() # Ensure PyAudio is initialized

        self.input_devices = []
        default_system_device_index: Optional[int] = None
        host_api_info = None
        default_host_api_index = 0 # Default to first API if detection fails

        try:
             # Try getting default host API info
             default_host_api_index = self.p.get_default_host_api_info()['index']
             host_api_info = self.p.get_host_api_info_by_index(default_host_api_index)
             default_system_device_index = host_api_info.get('defaultInputDevice')
             # PyAudio returns -1 if no default input device for the API
             if default_system_device_index == -1: default_system_device_index = None
             logger.info(f"Default Host API: {host_api_info.get('name')}, Default Input Device Index: {default_system_device_index}")
        except Exception as e:
            logger.warning(f"Could not get default device via Host API info: {e}. Will check all devices.")
            # Attempt to get default input device directly (might work on some systems)
            try:
                 default_info = self.p.get_default_input_device_info()
                 default_system_device_index = default_info['index']
                 default_host_api_index = default_info['hostApi'] # Use API of this default device
                 logger.info(f"Found default input device directly: {default_info.get('name')} (Index: {default_system_device_index}, API: {default_host_api_index})")
            except Exception as e2:
                 logger.warning(f"Could not get default input device info directly: {e2}. Falling back to first compatible.")


        device_count = self.p.get_device_count()
        logger.debug(f"Total audio devices found: {device_count}")
        selected_device_info = None # Store info of selected device

        # Iterate through devices, checking compatibility
        for i in range(device_count):
             try:
                device_info = self.p.get_device_info_by_index(i)
                # Check if device belongs to the relevant API and supports input
                if device_info.get('hostApi') == default_host_api_index and \
                   device_info.get('maxInputChannels', 0) >= self.channels:
                     # Now check format support
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
                             # Select if it's the system default or if we haven't selected one yet
                             if self.input_device_index is None: # Select the first compatible one initially
                                  self.input_device_index = device_info['index']
                                  selected_device_info = device_info
                             if default_system_device_index == i: # If this is the actual default, prioritize it
                                  self.input_device_index = i
                                  selected_device_info = device_info
                                  logger.info(f"Selecting system default input device: {device_info.get('name')} (Index: {i})")
                                  # Don't break, let loop finish logging others

                     except ValueError:
                          # This means PyAudio couldn't check support for this specific device index
                           logger.debug(f"Could not check format support for device {device_info.get('name')} (Index: {i}). Skipping.")
                     except OSError as pa_os_err:
                          # Sometimes is_format_supported raises OSError for invalid devices
                           logger.debug(f"Format check failed for device {i} (OSError): {pa_os_err}. Skipping.")

             except Exception as dev_e:
                 logger.warning(f"Could not query full device info for index {i}: {dev_e}")


        # Final check if a device was selected
        if self.input_device_index is None:
            logger.error(f"No compatible audio input devices found for Host API {default_host_api_index} supporting {self.sample_rate} Hz!")
            raise RuntimeError("No compatible audio input devices found.")
        else:
             # Ensure selected_device_info is populated if selection happened via default_system_device_index check
             if not selected_device_info:
                  try: selected_device_info = self.p.get_device_info_by_index(self.input_device_index)
                  except Exception: selected_device_info = {"name": f"Index {self.input_device_index}"}

             logger.info(f"Using input device: {selected_device_info.get('name', 'Unknown')} (Index: {self.input_device_index})")

    def setup_vad(self):
        """Initialize Voice Activity Detection."""
        logger.debug("Setting up VAD...")
        if not self.sample_rate: raise ValueError("Sample rate must be set before VAD setup.")
        self.vad = Vad(3) # Aggressiveness level 3 (0-3)
        # Ensure frame duration is valid for webrtcvad (10, 20, or 30 ms)
        if self.frame_duration not in [10, 20, 30]:
             logger.warning(f"Invalid VAD frame duration {self.frame_duration}ms. Setting to 30ms.")
             self.frame_duration = 30
        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        self.prev_frames_maxlen = int(self.prev_frames_duration * self.sample_rate / self.frame_length)
        self.min_silence_detections = int(self.silence_limit * 1000 / self.frame_duration)
        logger.debug(f"VAD setup: Frame Length={self.frame_length}, Buffer={self.prev_frames_maxlen} frames, Silence Chunks={self.min_silence_detections}")

    def set_transcript_callback(self, callback: Callable[[str, str], None]):
        """Set callback for transcript updates (text, source_type)."""
        self.transcript_callback = callback

    def start_background_listener(self):
        """Start background listening to keep rolling buffer updated."""
        if not self.p: logger.error("PyAudio not initialized in background listener."); return
        if self.input_device_index is None: logger.error("No input device selected for background listener."); return

        if self.background_stream is None and not self.is_background_listening:
            logger.debug("Starting background listener for rolling buffer")
            try:
                self.is_background_listening = True
                # Optional: Wait for espeak to finish? Can cause delays.
                # if is_espeak_running():
                #     logger.warning("Waiting for espeak before starting background listener...")
                #     # Add a sensible timeout
                #     for _ in range(30): # Wait up to 3 seconds
                #         if not is_espeak_running(): break
                #         time.sleep(0.1)
                #     else: logger.warning("Timeout waiting for espeak.")

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
                     # If stream immediately fails, reset state
                     logger.error("Background listener stream failed to start (not active).")
                     self.is_background_listening = False
                     self.background_stream = None
            except Exception as e:
                logger.error(f"Failed to start background listener: {e}", exc_info=True)
                self.background_stream = None
                self.is_background_listening = False

    def stop_background_listener(self):
        """Stop background listening."""
        if self.background_stream is not None:
            logger.debug("Stopping background listener")
            self.is_background_listening = False # Set flag first
            stream_to_close = self.background_stream
            self.background_stream = None # Prevent callback issues during close
            try:
                 if stream_to_close.is_active():
                     stream_to_close.stop_stream()
                 stream_to_close.close()
                 logger.debug("Background listener stream stopped and closed.")
            except Exception as e:
                 logger.error(f"Error closing background stream: {e}")
            # Ensure variable is cleared even if close fails
            # self.background_stream = None # Already done above

    def background_callback(self, in_data, frame_count, time_info, status):
        """Handle audio input for background listening (rolling buffer)."""
        if not self.is_background_listening: return (None, pyaudio.paComplete)
        if status: logger.warning(f"Background audio input status non-zero: {status}"); return (None, pyaudio.paContinue) # Log warning, but continue
        if is_espeak_running(): return (in_data, pyaudio.paContinue) # Skip if speaking

        try:
            # Store numpy arrays if that's what downstream expects
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            self.prev_frames.append(audio_chunk.copy())
            # Efficiently trim list to max length
            del self.prev_frames[:-self.prev_frames_maxlen]
        except Exception as e:
            logger.error(f"Error in background audio processing: {e}", exc_info=True)

        return (in_data, pyaudio.paContinue)

    # --- start/stop_quick_record are SYNCHRONOUS ---
    # They will be called via run_in_executor from the hotkey listener
    def start_quick_record(self):
        """Start recording for quick command. (Synchronous)"""
        logger.info("Starting quick recording")
        if not self.p: logger.error("PyAudio not initialized."); return False
        if self.input_device_index is None: logger.error("No input device selected."); return False

        if self.is_recording: logger.warning("Already recording."); return False # Prevent overlap

        self.is_recording = True
        self.stop_background_listener() # Stop background capture first
        self.current_audio = self.prev_frames.copy() # Start with buffer
        logger.debug(f"Added {len(self.current_audio)} frames from rolling buffer")

        if self.stream is not None: # Close existing stream if somehow open
            logger.warning("Quick record stream was already open? Closing it.")
            try: self.stream.close()
            except Exception: pass
            self.stream = None

        try:
            self.running = True # Enable callback processing
            if is_espeak_running(): logger.info("Espeak running during record start.")

            # Open the stream for recording
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
                 self.is_recording = False; self.stream = None; return False
        except Exception as e:
            logger.error(f"Failed to start quick record stream: {e}", exc_info=True)
            self.stream = None; self.is_recording = False; return False


    def stop_quick_record(self):
        """Stop recording and process the command. (Synchronous)"""
        if not self.is_recording:
             logger.debug("Stop quick record called but not recording.")
             # Ensure background listener restarts if needed
             if self.background_stream is None and not self.is_background_listening:
                  self.start_background_listener()
             return

        logger.info("Stopping quick recording")
        self.is_recording = False # Set flag immediately

        # Stop and close the stream
        if self.stream is not None:
            stream_to_close = self.stream
            self.stream = None # Clear stream variable
            self.running = False # Signal callback to stop processing data
            try:
                # Make sure stream is active before stopping
                if stream_to_close.is_active():
                     stream_to_close.stop_stream()
                stream_to_close.close()
                logger.debug("Quick record stream stopped and closed.")
            except Exception as e:
                 logger.error(f"Error stopping/closing quick record stream: {e}")


        # Process the collected audio
        if self.current_audio:
            logger.debug(f"Processing recorded audio of size: {len(self.current_audio)} chunks")
            try:
                # Check if all elements are numpy arrays before concatenating
                if all(isinstance(chunk, np.ndarray) for chunk in self.current_audio):
                    combined_audio = np.concatenate(self.current_audio)
                    # Schedule the async processing in the main loop
                    self._schedule_process_speech(combined_audio)
                else:
                    logger.error("Audio buffer corrupted: contains non-numpy elements.")
            except ValueError as e:
                 logger.error(f"Error concatenating audio chunks: {e}. Chunks: {[type(c).__name__ for c in self.current_audio]}")
            except Exception as e:
                 logger.error(f"Unexpected error during audio combination: {e}")
            finally:
                 self.current_audio = [] # Clear buffer regardless
        else:
             logger.info("No audio recorded during quick record session.")


        # Restart background listener after stopping and processing
        # Needs to be done carefully to avoid race conditions if stop is called rapidly
        # Maybe schedule the restart slightly later?
        # For now, direct start.
        self.start_background_listener()


    def quick_record_callback(self, in_data, frame_count, time_info, status):
        """Handle audio input during quick recording."""
        if not self.running or not self.is_recording: return (None, pyaudio.paComplete)
        if status: logger.warning(f"Quick record audio input status non-zero: {status}"); return (None, pyaudio.paContinue)
        if is_espeak_running(): return (in_data, pyaudio.paContinue)

        try:
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            self.current_audio.append(audio_chunk.copy())
        except Exception as e:
            logger.error(f"Error in quick record audio callback: {e}", exc_info=True)

        return (in_data, pyaudio.paContinue)


    def _schedule_process_speech(self, audio_data):
        """Schedules the async process_speech method in the main event loop."""
        if not self.loop or not self.loop.is_running():
             logger.error("Cannot schedule speech processing: Event loop not available/running.")
             return
        try:
            if inspect.iscoroutinefunction(self.process_speech):
                 # Use call_soon_threadsafe to schedule task creation from executor thread
                 self.loop.call_soon_threadsafe(
                     lambda: asyncio.create_task(self.process_speech(audio_data))
                 )
            else:
                 logger.error("process_speech is not an async function! Cannot schedule.")
        except Exception as e:
             logger.error(f"Error scheduling speech processing: {e}", exc_info=True)

    async def process_speech(self, audio_data):
        """Process speech data, execute commands, print, and speak results."""
        logger.debug("Processing speech data asynchronously...")
        if not self.whisper or not self.command_processor or not self.transcript_callback or not self.speak_func:
            logger.error("Cannot process speech: Missing component."); return

        try:
            text = await self.whisper.transcribe(audio_data)
            if text:
                logger.info(f"Transcribed text: {text}")
                self.transcript_callback(text, "Voice") # Print transcription

                # Process command and handle results
                async for result in self.command_processor.process_command(text):
                    logger.info(f"Command result: {result}")
                    # Print each yielded result to CLI
                    self.transcript_callback(f"{result}", "System")
                    # Speak the command result if it's not an error/suggestion
                    if result and not result.startswith("[Error:") and not result.startswith("Suggested command:"):
                        await self.speak_func(result) # Call async speak function
            else:
                 logger.info("Transcription returned no text.")
                 self.transcript_callback("...", "Voice") # Indicate silence

        except Exception as e:
             logger.error(f"Error during async speech processing: {e}", exc_info=True)
             self.transcript_callback(f"[Error processing speech: {e}]", "Error")


    def cleanup(self):
        """Clean up resources before shutdown"""
        logger.info("Cleaning up VoiceCommandSystem resources...")
        self.running = False # Stop callbacks first
        self.is_background_listening = False
        self.is_recording = False

        # Ensure streams are stopped and closed
        if self.stream:
            try:
                if self.stream.is_active(): self.stream.stop_stream()
                self.stream.close()
            except Exception as e: logger.debug(f"Exception closing main stream: {e}") # Debug level
            finally: self.stream = None

        if self.background_stream:
            try:
                if self.background_stream.is_active(): self.background_stream.stop_stream()
                self.background_stream.close()
            except Exception as e: logger.debug(f"Exception closing background stream: {e}") # Debug level
            finally: self.background_stream = None

        # Terminate PyAudio
        if self.p:
            try:
                logger.debug("Terminating PyAudio...")
                self.p.terminate()
                logger.debug("PyAudio terminated.")
            except Exception as e: logger.error(f"Error terminating PyAudio: {e}")
            finally: self.p = None

        logger.info("Voice command system resources released.")
