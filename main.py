#!/usr/bin/env python3

import numpy as np
import torch
import sys
import warnings
import logging
import sounddevice as sd
from webrtcvad import Vad
import queue
import asyncio
import threading
import time
from commands.command_processor import CommandProcessor
from speech.whisper_processor import WhisperProcessor

# Configure logging
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

class VoiceCommandSystem:
    def __init__(self):
        """Initialize the voice command system components."""
        print("Initializing voice command system...")
        self.setup_system()
        print("System initialization complete")

    def setup_system(self):
        """Set up all system components."""
        try:
            # Initialize Whisper
            self.whisper = WhisperProcessor()
            
            # Initialize VAD
            self.setup_vad()
            
            # Initialize command processor
            self.command_processor = CommandProcessor()
            
            # Set up audio queue
            self.audio_queue = queue.Queue()
            
        except Exception as e:
            print(f"Error during system setup: {e}")
            raise

    def setup_vad(self):
        """Initialize Voice Activity Detection components."""
        print("\nInitializing Voice Activity Detection...")
        self.vad = Vad(3)  # Aggressiveness level 3
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        
        # Speech detection state
        self.is_speaking = False
        self.current_audio = []
        self.last_speech_time = 0
        self.silence_threshold = 0.5  # seconds
        self.min_frames = 15  # Minimum frames for processing
        
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            print(f"[{i}] {dev['name']}")
        print(f"\nUsing default input device")

    def should_process_segment(self):
        """Check if enough silence time has passed to process the segment."""
        if not self.is_speaking:
            return False
        
        silence_duration = time.time() - self.last_speech_time
        return silence_duration >= self.silence_threshold

    def process_audio(self, indata, frames, time_info, status):
        """Process incoming audio data from the microphone."""
        if status:
            print(f"Audio input error: {status}")
            return

        try:
            # Convert input to proper format for VAD
            audio_chunk = np.frombuffer(indata, dtype=np.int16)
            is_speech = self.vad.is_speech(audio_chunk.tobytes(), self.sample_rate)

            if is_speech:
                self.last_speech_time = time.time()
                if not self.is_speaking:
                    print("\nSpeech detected")
                    self.is_speaking = True
                self.current_audio.append(audio_chunk.copy())
            else:
                if self.is_speaking and self.should_process_segment():
                    if len(self.current_audio) > self.min_frames:
                        print("Processing speech segment")
                        full_audio = np.concatenate(self.current_audio)
                        self.audio_queue.put(full_audio)
                    self.is_speaking = False
                    self.current_audio = []
                elif self.is_speaking:
                    # Continue collecting audio during short silences
                    self.current_audio.append(audio_chunk.copy())

        except Exception as e:
            print(f"Error processing audio chunk: {e}")

    async def process_command(self, text):
        """Process transcribed text as a command."""
        if not text:
            return
            
        print(f"Processing command: {text}")
        async for result in self.command_processor.process_command(text):
            print(f"Command result: {result}")

    def run(self):
        """Run the voice command system."""
        print("Starting audio stream...")
        
        async def audio_processor():
            while True:
                try:
                    # Get audio from queue
                    audio_data = await asyncio.to_thread(self.audio_queue.get)
                    
                    # Transcribe audio
                    text = await self.whisper.transcribe(audio_data)
                    
                    # Process any commands
                    if text:
                        await self.process_command(text)
                        
                except Exception as e:
                    print(f"Processing error: {e}")

        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Start audio processing in a separate thread
        processing_thread = threading.Thread(
            target=lambda: loop.run_until_complete(audio_processor()),
            daemon=True
        )
        processing_thread.start()

        # Start audio input stream
        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_length,
            dtype=np.int16,
            callback=self.process_audio
        ):
            print("\nSystem active - listening for commands")
            print("Available commands:")
            print("  - click <text>: Click text on screen")
            print("  - type <text>: Type text")
            print("  - computer <query>: Ask a question about highlighted text")
            
            try:
                while True:
                    sd.sleep(1000)  # Sleep to prevent high CPU usage
            except KeyboardInterrupt:
                print("\nStopping system")
                loop.stop()

if __name__ == "__main__":
    try:
        system = VoiceCommandSystem()
        system.run()
    except KeyboardInterrupt:
        print("\nSystem shutdown")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
