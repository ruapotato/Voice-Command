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

from webrtcvad import Vad
from commands.command_processor import CommandProcessor
from speech.whisper_processor import WhisperProcessor

logger = logging.getLogger(__name__)

# is_espeak_running function remains the same...
def is_espeak_running():
    """Check if espeak is currently running"""
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if proc.info['name'] == 'espeak' or \
               (proc.info['cmdline'] and proc.info['cmdline'] and 'espeak' in proc.info['cmdline'][0]):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


class VoiceCommandSystem:
    # <<< ADDED loop parameter >>>
    def __init__(self, loop: asyncio.AbstractEventLoop):
        """
        Initialize the voice command system components.
        Requires the main asyncio event loop.
        """
        logger.info("Initializing voice command system...")
        self.loop = loop # <<< STORED loop reference
        self.transcript_callback = None
        self.input_device_index = None
        self.command_processor: Optional[CommandProcessor] = None
        self.whisper: Optional[WhisperProcessor] = None
        self.p = None
        self.vad = None
        self.stream = None
        self.background_stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_background_listening = False
        self.current_audio = []
        self.prev_frames = []
        self.running = True

        try:
            self.setup_system()
            # Start background listener only *after* successful setup
            # Consider starting it via the loop for better async integration?
            # For now, direct start is okay if PyAudio handles threads safely.
            self.start_background_listener()
        except Exception as e:
             logger.error(f"Error during system setup: {e}", exc_info=True)
             raise

        logger.info("Voice system initialization complete.")

    def setup_system(self):
        """Set up all system components."""
        self.sample_rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16

        self.p = pyaudio.PyAudio()
        self._detect_audio_devices()

        # Assuming WhisperProcessor() is safe to call synchronously
        self.whisper = WhisperProcessor()

        self.setup_vad()

        # Command processor doesn't need the loop
        self.command_processor = CommandProcessor()

        self.chunk_size = self.frame_length

    # _detect_audio_devices remains the same
    def _detect_audio_devices(self):
        """Detect available audio input devices and select default"""
        logger.info("Detecting audio input devices...")
        if not self.p: self.p = pyaudio.PyAudio()

        self.input_devices = []
        default_device_index = None
        host_api_info = None
        try:
             default_host_api_index = self.p.get_default_host_api_info()['index']
             host_api_info = self.p.get_host_api_info_by_index(default_host_api_index)
             default_device_index = host_api_info.get('defaultInputDevice')
             # Ensure default device index is valid
             if default_device_index == -1: default_device_index = None
             logger.info(f"Default Host API: {host_api_info.get('name')}, Default Input Device Index: {default_device_index}")
        except Exception as e:
            logger.warning(f"Could not get default device via Host API: {e}")
            default_host_api_index = 0 # Fallback to checking first host API

        device_count = self.p.get_device_count()
        logger.debug(f"Total audio devices found: {device_count}")

        selected_device_info = None # Store info of selected device

        for i in range(device_count):
             try:
                device_info = self.p.get_device_info_by_index(i)
                # Check API match and input channels first
                if device_info.get('hostApi') == default_host_api_index and device_info.get('maxInputChannels', 0) > 0:
                     # Now check format support
                     try:
                         supported = self.p.is_format_supported(
                             self.sample_rate,
                             input_device=device_info['index'],
                             input_channels=self.channels,
                             input_format=self.format
                         )
                         if supported:
                             logger.debug(f"Compatible input device found: {device_info.get('name')} (Index: {i})")
                             self.input_devices.append(device_info)
                             # Select if it's the default or if we haven't selected one yet
                             if self.input_device_index is None: # Select the first compatible one initially
                                  self.input_device_index = device_info['index']
                                  selected_device_info = device_info
                             if default_device_index == i: # If this is the actual default, prioritize it
                                  self.input_device_index = i
                                  selected_device_info = device_info
                                  logger.info(f"Selecting default input device: {device_info.get('name')} (Index: {i})")
                                  # Don't break, let loop finish in case of logging issues later
                     except ValueError:
                          logger.debug(f"Device {device_info.get('name')} (Index: {i}) does not support format.")
             except Exception as dev_e:
                 logger.warning(f"Could not query device info for index {i}: {dev_e}")


        if self.input_device_index is None:
            logger.error("No compatible audio input devices found for the default host API!")
            # Optionally, could search other Host APIs here
            raise RuntimeError("No compatible audio input devices found.")
        else:
             logger.info(f"Using input device: {selected_device_info.get('name', 'Unknown')} (Index: {self.input_device_index})")


    # setup_vad remains the same
    def setup_vad(self):
        """Initialize Voice Activity Detection."""
        logger.debug("Setting up VAD...")
        self.vad = Vad(3)
        self.frame_duration = 30
        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        self.prev_frames_duration = 0.5
        self.prev_frames = []
        self.prev_frames_maxlen = int(self.prev_frames_duration * self.sample_rate / self.frame_length)
        self.silence_limit = 0.7
        self.min_silence_detections = int(self.silence_limit * 1000 / self.frame_duration)
        logger.debug(f"VAD setup: Frame Length={self.frame_length}, Buffer={self.prev_frames_maxlen} frames, Silence Chunks={self.min_silence_detections}")

    # set_transcript_callback remains the same
    def set_transcript_callback(self, callback):
        self.transcript_callback = callback

    # start_background_listener remains the same
    def start_background_listener(self):
        """Start background listening to keep rolling buffer updated."""
        if not self.p: logger.error("PyAudio not initialized in background listener."); return
        if self.input_device_index is None: logger.error("No input device selected for background listener."); return

        if self.background_stream is None and not self.is_background_listening:
            logger.debug("Starting background listener for rolling buffer")
            try:
                self.is_background_listening = True
                if is_espeak_running():
                     logger.warning("Waiting for espeak to finish before starting background listener...")
                     for _ in range(50): # Wait up to 5 seconds
                          if not is_espeak_running(): break
                          time.sleep(0.1)
                     else: logger.error("Timeout waiting for espeak.")

                self.background_stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.frame_length,
                    stream_callback=self.background_callback
                )
                if self.background_stream.is_active(): logger.debug("Background listener stream started successfully.")
                else: logger.error("Background listener stream failed to start."); self.is_background_listening = False; self.background_stream = None
            except Exception as e:
                logger.error(f"Failed to start background listener: {e}", exc_info=True)
                self.background_stream = None; self.is_background_listening = False

    # stop_background_listener remains the same
    def stop_background_listener(self):
        """Stop background listening."""
        if self.background_stream is not None:
            logger.debug("Stopping background listener")
            self.is_background_listening = False
            try:
                 if self.background_stream.is_active(): self.background_stream.stop_stream()
                 self.background_stream.close()
            except Exception as e: logger.error(f"Error closing background stream: {e}")
            finally: self.background_stream = None

    # background_callback remains the same
    def background_callback(self, in_data, frame_count, time_info, status):
        """Handle audio input for background listening (rolling buffer)."""
        if not self.is_background_listening: return (None, pyaudio.paComplete)
        if status: logger.error(f"Background audio input error: {status}"); return (None, pyaudio.paContinue)
        if is_espeak_running(): return (in_data, pyaudio.paContinue)
        try:
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            self.prev_frames.append(audio_chunk.copy())
            if len(self.prev_frames) > self.prev_frames_maxlen: self.prev_frames.pop(0)
        except Exception as e: logger.error(f"Error in background audio processing: {e}", exc_info=True)
        return (in_data, pyaudio.paContinue)

    # start_quick_record remains the same (sync)
    def start_quick_record(self):
        """Start recording for quick command. (Synchronous)"""
        logger.info("Starting quick recording")
        if not self.p: logger.error("PyAudio not initialized in start_quick_record."); return
        if self.input_device_index is None: logger.error("No input device selected for quick record."); return

        self.is_recording = True
        if self.background_stream is not None: self.stop_background_listener()
        self.current_audio = self.prev_frames.copy()
        logger.debug(f"Added {len(self.current_audio)} frames from rolling buffer")

        if self.stream is None:
            try:
                self.running = True
                if is_espeak_running(): logger.info("Espeak running, recording might capture feedback.")
                self.stream = self.p.open(
                    format=self.format, channels=self.channels, rate=self.sample_rate,
                    input=True, input_device_index=self.input_device_index,
                    frames_per_buffer=self.frame_length, stream_callback=self.quick_record_callback
                )
                if self.stream.is_active(): logger.debug("Quick record stream started.")
                else: logger.error("Quick record stream failed to start."); self.is_recording = False; self.stream = None
            except Exception as e:
                logger.error(f"Failed to start quick record stream: {e}", exc_info=True)
                self.stream = None; self.is_recording = False

    # stop_quick_record remains the same (sync)
    def stop_quick_record(self):
        """Stop recording and process the command. (Synchronous)"""
        logger.info("Stopping quick recording")
        self.is_recording = False
        if self.stream is not None:
            self.running = False
            try:
                if self.stream.is_active(): self.stream.stop_stream()
                self.stream.close()
            except Exception as e: logger.error(f"Error closing quick record stream: {e}")
            finally: self.stream = None

        if self.current_audio:
            logger.debug(f"Processing recorded audio of size: {len(self.current_audio)} chunks")
            try:
                if all(isinstance(chunk, np.ndarray) for chunk in self.current_audio):
                    combined_audio = np.concatenate(self.current_audio)
                    # <<< FIX: Use _schedule_process_speech >>>
                    self._schedule_process_speech(combined_audio)
                else: logger.error("Error: current_audio contains non-numpy array elements.")
            except ValueError as e: logger.error(f"Error concatenating audio chunks: {e}.")
            except Exception as e: logger.error(f"Unexpected error during audio combination: {e}")
            finally: self.current_audio = []
        else: logger.info("No audio recorded during quick record session.")

        self.start_background_listener()

    # quick_record_callback remains the same
    def quick_record_callback(self, in_data, frame_count, time_info, status):
        """Handle audio input during quick recording."""
        if not self.running or not self.is_recording: return (None, pyaudio.paComplete)
        if status: logger.error(f"Quick record audio input error: {status}"); return (None, pyaudio.paContinue)
        if is_espeak_running(): return (in_data, pyaudio.paContinue)
        try:
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            self.current_audio.append(audio_chunk.copy())
        except Exception as e: logger.error(f"Error in quick record audio callback: {e}", exc_info=True)
        return (in_data, pyaudio.paContinue)

    # --- _schedule_process_speech (Corrected) ---
    def _schedule_process_speech(self, audio_data):
        """Schedules the async process_speech method in the main event loop."""
        # <<< FIX: Check if self.loop exists >>>
        if not self.loop or not self.loop.is_running():
             logger.error("Cannot schedule speech processing: Event loop not available or not running.")
             return

        try:
            # Ensure process_speech is async before creating task
            if inspect.iscoroutinefunction(self.process_speech):
                 # <<< FIX: Use self.loop.call_soon_threadsafe >>>
                 self.loop.call_soon_threadsafe(
                     lambda: asyncio.create_task(self.process_speech(audio_data))
                 )
            else:
                 logger.error("process_speech is not an async function! Cannot schedule.")
        except Exception as e:
             logger.error(f"Error scheduling speech processing: {e}", exc_info=True)

    # process_speech remains the same (async)
    async def process_speech(self, audio_data):
        """Process speech data and execute commands. (Async)"""
        logger.debug("Processing speech data asynchronously...")
        if not self.whisper: logger.error("Whisper processor not initialized."); return
        if not self.command_processor: logger.error("Command processor not initialized."); return
        if not self.transcript_callback: logger.warning("Transcript callback not set."); return

        try:
            # Assuming self.whisper.transcribe is async
            text = await self.whisper.transcribe(audio_data)
            if text:
                logger.info(f"Transcribed text: {text}")
                self.transcript_callback(text, "Voice")
                # Assuming self.command_processor.process_command is async generator
                async for result in self.command_processor.process_command(text):
                    logger.info(f"Command result: {result}")
                    self.transcript_callback(f"{result}", "System")
            else:
                 logger.info("Transcription returned no text.")
                 self.transcript_callback("...", "Voice")
        except Exception as e:
             logger.error(f"Error during async speech processing: {e}", exc_info=True)
             if self.transcript_callback:
                  self.transcript_callback(f"[Error processing speech: {e}]", "Error")

    # cleanup remains the same
    def cleanup(self):
        """Clean up resources before shutdown"""
        logger.info("Cleaning up VoiceCommandSystem resources...")
        self.running = False
        self.is_background_listening = False
        self.is_recording = False

        # Close streams safely
        for stream in [self.stream, self.background_stream]:
             if stream:
                  try:
                       if stream.is_active(): stream.stop_stream()
                       stream.close()
                  except Exception as e: logger.error(f"Error closing stream: {e}")
        self.stream = None
        self.background_stream = None

        # Terminate PyAudio
        if self.p:
            try: self.p.terminate()
            except Exception as e: logger.error(f"Error terminating PyAudio: {e}")
            self.p = None
        logger.info("Voice command system resources released.")
