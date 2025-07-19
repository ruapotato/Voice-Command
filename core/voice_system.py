# core/voice_system.py
import numpy as np
import warnings
import logging
import pyaudio
import queue
import asyncio
import threading
import psutil
import time
from typing import Optional, Callable, Awaitable, Any, Coroutine

from webrtcvad import Vad
from speech.whisper_processor import ParakeetProcessor

logger = logging.getLogger(__name__)

def is_espeak_running():
    """Check if espeak is currently running."""
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
        logger.info("Initializing voice command system...")
        self.loop = loop
        self.speak_func = speak_func
        self.command_trigger_func = command_trigger_func
        self.transcript_callback: Optional[Callable[[str, str], None]] = None
        self.input_device_index: Optional[int] = None

        self.asr_processor: Optional[ParakeetProcessor] = None
        self.p: Optional[pyaudio.PyAudio] = None
        self.vad: Optional[Vad] = None
        
        # --- REFACTORED FOR THREAD-SAFE AUDIO HANDLING ---
        self.background_stream: Optional[pyaudio.Stream] = None
        self.quick_record_stream: Optional[pyaudio.Stream] = None

        self.background_queue: queue.Queue = queue.Queue()
        self.quick_record_queue: queue.Queue = queue.Queue()

        self.background_worker_thread: Optional[threading.Thread] = None
        self.quick_record_worker_thread: Optional[threading.Thread] = None

        self.is_background_listening = threading.Event()
        self.is_quick_recording = threading.Event()
        
        self.current_audio: list[np.ndarray] = []
        self.prev_frames: list[np.ndarray] = []
        # --- END REFACTOR ---

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
            logger.error(f"Critical error during VoiceCommandSystem setup: {e}", exc_info=True)
            self.cleanup()
            raise

        logger.info("Voice system initialization complete.")

    def setup_system(self):
        logger.debug("Setting up system components...")
        self.p = pyaudio.PyAudio()
        self._detect_audio_devices()
        self.asr_processor = ParakeetProcessor()
        self.setup_vad()
        self.chunk_size = self.frame_length
        logger.debug("System components setup finished.")

    def _detect_audio_devices(self):
        # This extensive device detection logic is good. No changes needed here.
        logger.info("Detecting audio input devices...")
        if not self.p:
            logger.error("PyAudio not initialized before _detect_audio_devices.")
            self.p = pyaudio.PyAudio()

        self.input_devices = []
        default_system_device_index: Optional[int] = None
        host_api_info = None
        default_host_api_index = 0

        try:
            default_host_api_info_dict = self.p.get_default_host_api_info()
            default_host_api_index = default_host_api_info_dict['index']
            host_api_info = self.p.get_host_api_info_by_index(default_host_api_index)
            default_system_device_index = host_api_info.get('defaultInputDevice')
            if default_system_device_index == -1:
                default_system_device_index = None
            logger.info(f"Default Host API: {host_api_info.get('name')}, Default System Input Device Index: {default_system_device_index}")
        except Exception as e:
            logger.warning(f"Could not get default device via Host API info: {e}. Will iterate.")
            try:
                default_input_device_info = self.p.get_default_input_device_info()
                default_system_device_index = default_input_device_info['index']
                default_host_api_index = default_input_device_info['hostApi']
                logger.info(f"Found default input device directly: {default_input_device_info.get('name')} (Index: {default_system_device_index})")
            except Exception as e2:
                logger.warning(f"Could not get default input device info directly: {e2}. Will iterate all devices.")
                default_system_device_index = None

        device_count = self.p.get_device_count()
        selected_device_info = None

        for i in range(device_count):
            try:
                device_info = self.p.get_device_info_by_index(i)
                is_on_preferred_api = (device_info.get('hostApi') == default_host_api_index)
                has_input_channels = device_info.get('maxInputChannels', 0) >= self.channels

                if has_input_channels:
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
                                self.input_devices.append(device_info)
                                if default_system_device_index == i:
                                    self.input_device_index = i
                                    selected_device_info = device_info
                                    logger.info(f"Selecting system default input device: {device_info.get('name')} (Index: {i})")
                                    break
                                elif self.input_device_index is None:
                                    self.input_device_index = device_info['index']
                                    selected_device_info = device_info
                        except ValueError:
                            pass # Format not supported
            except Exception as dev_e:
                logger.warning(f"Could not query full device info for index {i}: {dev_e}")
        
        if self.input_device_index is None:
            final_error_msg = f"No compatible audio input devices found."
            logger.error(final_error_msg)
            raise RuntimeError(final_error_msg)
        else:
            if not selected_device_info:
                selected_device_info = self.p.get_device_info_by_index(self.input_device_index)
            logger.info(f"Using audio input device: {selected_device_info.get('name', 'N/A')} (Index: {self.input_device_index})")

    def setup_vad(self):
        logger.debug("Setting up VAD...")
        self.vad = Vad(3)
        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        self.prev_frames_maxlen = int(self.prev_frames_duration * self.sample_rate / self.frame_length)
        self.min_silence_detections = int(self.silence_limit * 1000 / self.frame_duration)

    def set_transcript_callback(self, callback: Callable[[str, str], None]):
        self.transcript_callback = callback

    # --- WORKER THREAD METHODS ---

    def _background_worker(self):
        """CONSUMER for the background listener. Processes audio from its queue."""
        while self.is_background_listening.is_set():
            try:
                in_data = self.background_queue.get(timeout=0.5)
                if is_espeak_running():
                    continue
                audio_chunk = np.frombuffer(in_data, dtype=np.int16)
                self.prev_frames.append(audio_chunk.copy())
                if len(self.prev_frames) > self.prev_frames_maxlen:
                    self.prev_frames.pop(0)
            except queue.Empty:
                continue # This is normal, just loop again
            except Exception as e:
                logger.error(f"Error in background worker: {e}", exc_info=True)

    def _quick_record_worker(self):
        """CONSUMER for the quick recorder. Processes audio from its queue."""
        while self.is_quick_recording.is_set() or not self.quick_record_queue.empty():
            try:
                in_data = self.quick_record_queue.get(timeout=0.5)
                if is_espeak_running():
                    continue
                audio_chunk = np.frombuffer(in_data, dtype=np.int16)
                self.current_audio.append(audio_chunk.copy())
            except queue.Empty:
                if not self.is_quick_recording.is_set():
                    break # Exit if recording is stopped and queue is empty
            except Exception as e:
                logger.error(f"Error in quick record worker: {e}", exc_info=True)

    # --- CALLBACK METHODS (PRODUCERS) ---

    def background_callback(self, in_data, frame_count, time_info, status_flags):
        """PRODUCER: Puts background audio data into a queue. Must be fast."""
        if status_flags:
            logger.warning(f"Background audio input status flags non-zero: {status_flags}")
        self.background_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def quick_record_callback(self, in_data, frame_count, time_info, status_flags):
        """PRODUCER: Puts quick record audio data into a queue. Must be fast."""
        if status_flags:
            logger.warning(f"Quick record audio input status flags non-zero: {status_flags}")
        self.quick_record_queue.put(in_data)
        return (None, pyaudio.paContinue)

    # --- STREAM CONTROL METHODS ---

    def start_background_listener(self):
        if self.is_background_listening.is_set():
            return
        logger.debug("Starting background listener...")
        self.is_background_listening.set()
        
        self.background_worker_thread = threading.Thread(target=self._background_worker)
        self.background_worker_thread.start()

        self.background_stream = self.p.open(
            format=self.format, channels=self.channels, rate=self.sample_rate,
            input=True, input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size * 2, # A slightly larger buffer is safe
            stream_callback=self.background_callback
        )
        logger.debug("Background listener stream started.")

    def stop_background_listener(self):
        if not self.is_background_listening.is_set():
            return
        logger.debug("Stopping background listener...")
        self.is_background_listening.clear()

        if self.background_stream:
            self.background_stream.stop_stream()
            self.background_stream.close()
            self.background_stream = None
        
        if self.background_worker_thread:
            self.background_worker_thread.join()
            self.background_worker_thread = None
        logger.debug("Background listener stopped.")

    def start_quick_record(self):
        if self.is_quick_recording.is_set():
            return False
        logger.info("Attempting to start quick recording...")
        self.stop_background_listener()
        
        self.is_quick_recording.set()
        self.current_audio = list(self.prev_frames)
        logger.debug(f"Initialized quick recording with {len(self.current_audio)} frames.")

        self.quick_record_worker_thread = threading.Thread(target=self._quick_record_worker)
        self.quick_record_worker_thread.start()

        self.quick_record_stream = self.p.open(
            format=self.format, channels=self.channels, rate=self.sample_rate,
            input=True, input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.quick_record_callback
        )
        logger.info("Quick recording stream started successfully.")
        return True

    def stop_quick_record(self):
        if not self.is_quick_recording.is_set():
            return
        logger.info("Stopping quick recording...")
        self.is_quick_recording.clear()

        if self.quick_record_stream:
            self.quick_record_stream.stop_stream()
            self.quick_record_stream.close()
            self.quick_record_stream = None

        if self.quick_record_worker_thread:
            self.quick_record_worker_thread.join() # Wait for worker to process all data
            self.quick_record_worker_thread = None

        if self.current_audio:
            try:
                combined_audio_data = np.concatenate(self.current_audio)
                self._schedule_process_speech(combined_audio_data)
            except ValueError as e:
                logger.error(f"Error concatenating audio chunks: {e}", exc_info=True)
            finally:
                self.current_audio = []
        else:
            logger.info("No audio was captured.")
            if self.transcript_callback:
                self.loop.call_soon_threadsafe(self.transcript_callback, "...", "Voice")
        
        self.start_background_listener() # Restart background listener
        logger.debug("Background listener restarted.")

    # --- ASYNC SPEECH PROCESSING ---
    
    def _schedule_process_speech(self, audio_data: np.ndarray):
        """Schedules the async process_speech method in the main event loop."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(
                lambda: self.loop.create_task(self.process_speech(audio_data))
            )

    async def process_speech(self, audio_data: np.ndarray):
        """Transcribes speech and triggers command processing."""
        # This async logic is well-structured. No changes needed here.
        if not self.asr_processor or not self.command_trigger_func:
            logger.error("ASR processor or command trigger not initialized.")
            return

        try:
            transcribed_text = await self.asr_processor.transcribe(audio_data)
            
            if self.transcript_callback:
                text_to_print = transcribed_text if transcribed_text else "..."
                self.transcript_callback(text_to_print, "Voice")

            if transcribed_text:
                logger.info(f"Transcription successful: '{transcribed_text}'. Triggering command.")
                await self.command_trigger_func(transcribed_text)
            else:
                logger.info("Transcription returned no text.")
        except Exception as e:
            logger.error(f"Error during async speech processing: {e}", exc_info=True)
            if self.transcript_callback:
                self.transcript_callback("[Error processing speech]", "Error")

    def cleanup(self):
        """Cleans up all resources."""
        logger.info("Cleaning up VoiceCommandSystem resources...")
        self.stop_quick_record()
        self.stop_background_listener()
        
        if self.p:
            self.p.terminate()
            self.p = None
        logger.info("Voice command system resources released.")
