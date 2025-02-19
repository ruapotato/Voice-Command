import numpy as np
import torch
import warnings
import logging
import sounddevice as sd
from webrtcvad import Vad
import queue
import asyncio
import threading
from commands.command_processor import CommandProcessor
from speech.whisper_processor import WhisperProcessor

logger = logging.getLogger(__name__)

class VoiceCommandSystem:
    def __init__(self):
        """Initialize the voice command system components."""
        logger.info("Initializing voice command system...")
        self.transcript_callback = None
        self.window = None
        self.setup_system()
        logger.info("System initialization complete")

    def setup_system(self):
        """Set up all system components."""
        try:
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
            
        except Exception as e:
            logger.error(f"Error during system setup: {e}", exc_info=True)
            raise

    def set_window(self, window):
        """Set the window reference"""
        self.window = window
        self.command_processor.set_window(window)

    def setup_vad(self):
        """Initialize Voice Activity Detection."""
        logger.debug("Setting up VAD...")
        self.vad = Vad(3)  # Aggressiveness level 3
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        logger.debug("VAD setup complete")

    def set_transcript_callback(self, callback):
        """Set callback for transcript updates"""
        self.transcript_callback = callback

    def start_listening(self):
        """Start continuous listening mode."""
        logger.info("Starting continuous listening mode")
        if self.stream is None:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.frame_length,
                dtype=np.int16,
                callback=self.process_audio
            )
            self.stream.start()
            logger.debug("Audio stream started")

    def stop_listening(self):
        """Stop continuous listening mode."""
        logger.info("Stopping listening mode")
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            logger.debug("Audio stream stopped")

    def start_quick_record(self):
        """Start recording for quick command."""
        logger.info("Starting quick recording")
        self.is_recording = True
        self.current_audio = []
        
        if self.stream is None:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.frame_length,
                dtype=np.int16,
                callback=self.process_quick_record
            )
            self.stream.start()
            logger.debug("Quick record stream started")

    def stop_quick_record(self):
        """Stop recording and process the command."""
        logger.info("Stopping quick recording")
        self.is_recording = False
        
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        if self.current_audio:
            logger.debug(f"Processing recorded audio of size: {len(self.current_audio)}")
            combined_audio = np.concatenate(self.current_audio)
            self._process_speech_sync(combined_audio)
            self.current_audio = []

    def process_quick_record(self, indata, frames, time_info, status):
        """Handle audio input during quick recording."""
        if status:
            logger.error(f"Audio input error: {status}")
            return
            
        if not self.is_recording:
            return
            
        audio_chunk = np.frombuffer(indata, dtype=np.int16)
        self.current_audio.append(audio_chunk.copy())
        logger.debug(f"Recorded chunk size: {len(audio_chunk)}")

    def process_audio(self, indata, frames, time_info, status):
        """Process incoming audio data with VAD."""
        if status:
            logger.error(f"Audio input error: {status}")
            return

        try:
            audio_chunk = np.frombuffer(indata, dtype=np.int16)
            is_speech = self.vad.is_speech(audio_chunk.tobytes(), self.sample_rate)

            if is_speech:
                logger.debug("Speech detected")
                self.audio_queue.put(audio_chunk.copy())
            else:
                # Process accumulated audio if we have enough
                if not self.audio_queue.empty():
                    full_audio = []
                    while not self.audio_queue.empty():
                        full_audio.append(self.audio_queue.get())
                    if len(full_audio) > 15:  # Minimum frames threshold
                        logger.debug("Processing speech segment")
                        combined_audio = np.concatenate(full_audio)
                        self._process_speech_sync(combined_audio)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)

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
