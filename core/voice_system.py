import numpy as np
import torch
import warnings
import logging
import pyaudio
import queue
import asyncio
import threading
import psutil
from webrtcvad import Vad
from commands.command_processor import CommandProcessor
from speech.whisper_processor import WhisperProcessor

logger = logging.getLogger(__name__)

def is_espeak_running():
    """Check if espeak is currently running"""
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if proc.info['name'] == 'espeak' or (proc.info['cmdline'] and 'espeak' in proc.info['cmdline'][0]):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

class VoiceCommandSystem:
    def __init__(self):
        """Initialize the voice command system components."""
        logger.info("Initializing voice command system...")
        self.transcript_callback = None
        self.window = None
        self.input_device_index = None
        self.setup_system()
        logger.info("System initialization complete")

    def setup_system(self):
        """Set up all system components."""
        try:
            # Audio parameters - define these first so setup_vad can use them
            self.sample_rate = 16000
            self.channels = 1
            self.format = pyaudio.paInt16
            
            # Initialize audio components
            self.p = pyaudio.PyAudio()
            self._detect_audio_devices()
            
            # Initialize Whisper
            self.whisper = WhisperProcessor()
            
            # Initialize VAD
            self.setup_vad()
            
            # Initialize command processor with window reference
            self.command_processor = CommandProcessor(window=self.window)
            
            # Set up audio queue and recording state
            self.audio_queue = queue.Queue()
            self.is_recording = False
            self.current_audio = []
            self.stream = None
            self.running = True
            
            # Set chunk size from VAD
            self.chunk_size = self.frame_length
            
        except Exception as e:
            logger.error(f"Error during system setup: {e}", exc_info=True)
            raise

    def _detect_audio_devices(self):
        """Detect available audio input devices and select default"""
        logger.info("Detecting audio input devices...")
        
        self.input_devices = []
        default_device_index = None
        
        # Enumerate all audio devices
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            
            # Check if this is an input device
            if device_info['maxInputChannels'] > 0:
                self.input_devices.append(device_info)
                logger.info(f"Found input device: {device_info['name']} (Index: {i})")
                
                # Check if this is the default input device
                if device_info.get('isDefaultInput', False) or 'default' in device_info['name'].lower():
                    default_device_index = i
                    logger.info(f"Selected default input device: {device_info['name']}")
        
        # If we found devices, set the default or first one
        if self.input_devices:
            if default_device_index is not None:
                self.input_device_index = default_device_index
            else:
                self.input_device_index = self.input_devices[0]['index']
                logger.info(f"No default device found, using first available: {self.input_devices[0]['name']}")
        else:
            logger.error("No input devices found!")
            
        logger.info(f"Using device index: {self.input_device_index}")

    def set_input_device(self, device_index):
        """Set the audio input device by index"""
        if self.stream:
            self.stop_listening()
            
        self.input_device_index = device_index
        logger.info(f"Changed input device to index: {device_index}")
        
        # Restart the stream if we were already listening
        if self.is_recording:
            self.start_listening()

    def get_input_devices(self):
        """Return a list of available input devices"""
        return self.input_devices

    def set_window(self, window):
        """Set the window reference"""
        self.window = window
        self.command_processor.set_window(window)

    def setup_vad(self):
        """Initialize Voice Activity Detection."""
        logger.debug("Setting up VAD...")
        self.vad = Vad(3)  # Aggressiveness level 3
        self.frame_duration = 30  # ms
        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        
        # Buffer to store audio before speech is detected
        self.prev_frames_duration = 0.5  # seconds
        self.prev_frames = []
        self.prev_frames_maxlen = int(self.prev_frames_duration * self.sample_rate / self.frame_length)
        
        # Silence detection parameters
        self.silence_limit = 0.7  # seconds
        self.min_silence_detections = 3  # minimum number of silent chunks
        
        logger.debug("VAD setup complete")

    def set_transcript_callback(self, callback):
        """Set callback for transcript updates"""
        self.transcript_callback = callback

    def start_listening(self):
        """Start continuous listening mode."""
        logger.info("Starting continuous listening mode")
        if self.stream is None:
            try:
                self.running = True
                
                # Check for espeak before starting
                if is_espeak_running():
                    logger.warning("Waiting for espeak to finish...")
                    while is_espeak_running():
                        import time
                        time.sleep(0.1)
                
                self.stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.frame_length,
                    stream_callback=self.audio_callback
                )
                self.is_recording = True
                logger.debug("Audio stream started")
            except Exception as e:
                logger.error(f"Failed to start audio stream: {e}", exc_info=True)
                self.stream = None
                raise

    def stop_listening(self):
        """Stop continuous listening mode."""
        logger.info("Stopping listening mode")
        if self.stream is not None:
            self.running = False
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            logger.debug("Audio stream stopped")

    def start_quick_record(self):
        """Start recording for quick command."""
        logger.info("Starting quick recording")
        self.is_recording = True
        self.current_audio = []
        
        if self.stream is None:
            try:
                self.running = True
                
                # Check for espeak before starting
                if is_espeak_running():
                    logger.info("Espeak is running, will ignore audio input")
                
                self.stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.frame_length,
                    stream_callback=self.quick_record_callback
                )
                logger.debug("Quick record stream started")
            except Exception as e:
                logger.error(f"Failed to start quick record: {e}", exc_info=True)
                self.stream = None
                raise

    def stop_quick_record(self):
        """Stop recording and process the command."""
        logger.info("Stopping quick recording")
        self.is_recording = False
        
        if self.stream is not None:
            self.running = False
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.current_audio:
            logger.debug(f"Processing recorded audio of size: {len(self.current_audio)}")
            combined_audio = np.concatenate(self.current_audio)
            self._process_speech_sync(combined_audio)
            self.current_audio = []

    def quick_record_callback(self, in_data, frame_count, time_info, status):
        """Handle audio input during quick recording."""
        if status:
            logger.error(f"Audio input error: {status}")
        
        if not self.running or not self.is_recording:
            return (None, pyaudio.paComplete)
            
        # Check for espeak before processing audio
        if is_espeak_running():
            return (in_data, pyaudio.paContinue)
            
        audio_chunk = np.frombuffer(in_data, dtype=np.int16)
        self.current_audio.append(audio_chunk.copy())
        logger.debug(f"Recorded chunk size: {len(audio_chunk)}")
        
        return (in_data, pyaudio.paContinue)

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Process incoming audio data with VAD."""
        if status:
            logger.error(f"Audio input error: {status}")
        
        if not self.running:
            return (None, pyaudio.paComplete)
            
        # Check for espeak before processing audio
        if is_espeak_running():
            # Skip processing while espeak is running
            return (in_data, pyaudio.paContinue)

        try:
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            
            # Convert bytes to expected format for VAD
            is_speech = self.vad.is_speech(audio_chunk.tobytes(), self.sample_rate)
            
            if is_speech:
                logger.debug("Speech detected")
                # Add to speech queue and keep track of previous frames
                self.audio_queue.put(audio_chunk.copy())
            else:
                # If we were collecting speech, add this silent chunk too
                if not self.audio_queue.empty():
                    self.audio_queue.put(audio_chunk.copy())
                    
                    # Process if we have enough silent frames after speech
                    # This allows us to capture a complete phrase
                    silent_frames = 0
                    while not self.audio_queue.empty() and silent_frames < self.min_silence_detections:
                        full_audio = []
                        queue_size = self.audio_queue.qsize()
                        for _ in range(queue_size):
                            full_audio.append(self.audio_queue.get())
                        
                        # Add silent frames count from the end of audio
                        for i in range(min(self.min_silence_detections, len(full_audio))):
                            check_chunk = full_audio[-(i+1)]
                            is_silent = not self.vad.is_speech(check_chunk.tobytes(), self.sample_rate)
                            if is_silent:
                                silent_frames += 1
                            else:
                                break
                        
                        # If we have enough silent frames or a minimum speech duration
                        if silent_frames >= self.min_silence_detections or len(full_audio) > (self.sample_rate / self.frame_length) * 2:
                            logger.debug("Processing speech segment")
                            combined_audio = np.concatenate(full_audio)
                            self._process_speech_sync(combined_audio)
                            break
                        else:
                            # Put frames back if we don't have enough silence yet
                            for frame in full_audio:
                                self.audio_queue.put(frame)
                else:
                    # Store some frames before speech for context
                    self.prev_frames.append(audio_chunk.copy())
                    if len(self.prev_frames) > self.prev_frames_maxlen:
                        self.prev_frames.pop(0)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)

        return (in_data, pyaudio.paContinue)

    def _process_speech_sync(self, audio_data):
        """Synchronous wrapper for processing speech."""
        try:
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.process_speech(audio_data))
                loop.close()

            # Run in thread
            thread = threading.Thread(target=run_async)
            thread.daemon = True
            thread.start()
        except Exception as e:
            logger.error(f"Error in speech processing: {e}", exc_info=True)
        return False  # Important for GLib.idle_add

    async def process_speech(self, audio_data):
        """Process speech data and execute commands."""
        logger.debug("Processing speech data")
        text = await self.whisper.transcribe(audio_data)
        if text:
            logger.info(f"Transcribed text: {text}")
            # Send transcript to UI
            if self.transcript_callback:
                self.transcript_callback(text)
            # Process as command
            await self.process_command(text)

    async def process_command(self, text):
        """Process a command string."""
        logger.info(f"Processing command: {text}")
        async for result in self.command_processor.process_command(text):
            logger.info(f"Command result: {result}")
            # Send result to UI if callback exists
            if self.transcript_callback:
                self.transcript_callback(f"Result: {result}", "System")

    def cleanup(self):
        """Clean up resources before shutdown"""
        self.running = False
        if self.stream:
            self.stop_listening()
        
        # Terminate PyAudio
        self.p.terminate()
        logger.info("Voice command system resources released")
