"""Command processor for voice commands."""
from dataclasses import dataclass, field
from typing import Dict, Any, AsyncGenerator, Callable, Optional, Tuple
import subprocess
import pyautogui
import pytesseract
import requests
import json
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image

@dataclass(frozen=True)
class Command:
    """A command with its state and execution function."""
    name: str
    aliases: list[str]  # Add aliases for command variations
    description: str
    execute: Callable
    state: Dict[str, bool] = field(default_factory=lambda: {'is_running': False})

    @property
    def is_active(self) -> bool:
        """Check if the command is currently running."""
        return self.state['is_running']
    
    @asynccontextmanager
    async def running(self):
        """Context manager for command execution state."""
        self.state['is_running'] = True
        try:
            yield
        finally:
            self.state['is_running'] = False

class CommandProcessor:
    def __init__(self):
        """Initialize the command processor."""
        self.commands = {
            "click": Command(
                "click",
                [],
                "Click text or buttons on screen",
                self._handle_click
            ),
            "type": Command(
                "type",
                ["type in"],  # Add "type in" as an alias
                "Type text using keyboard",
                self._handle_type
            ),
            "computer": Command(
                "computer",
                [],
                "Ask questions about highlighted text",
                self._handle_computer
            ),
            "read": Command(
                "read",
                ["reed", "red"],  # Add sound-alike variations
                "Read highlighted text aloud",
                self._handle_read
            )
        }
        self.espeak_config = "-ven+f3 -k5 -s150"  # Speech configuration
        print("Command processor initialized with:", list(self.commands.keys()))

    def parse_command(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse the voice command into command type and arguments."""
        text = text.strip()
        
        # Handle capitalization variations
        lower_text = text.lower()

        # Handle "type in" variations and cleanup
        if lower_text.startswith("type in"):
            args = text[7:].strip().strip(',').strip()  # Remove extra commas and spaces
            return "type", args
            
        if lower_text.startswith("type"):
            args = text[4:].strip().strip(',').strip()  # Remove extra commas and spaces
            return "type", args

        # Check for sound-alike commands
        for cmd, command_obj in self.commands.items():
            # Check main command name
            if lower_text.startswith(cmd):
                args = text[len(cmd):].strip()
                return cmd, args
            
            # Check aliases
            for alias in command_obj.aliases:
                if lower_text.startswith(alias):
                    args = text[len(alias):].strip()
                    return cmd, args

        return None, None

    async def _handle_click(self, text: str) -> str:
        """Handle click commands by finding and clicking matching text on screen."""
        try:
            print(f"Searching for text: '{text}'")
            screenshot = pyautogui.screenshot()
            
            # Configure Tesseract for better accuracy
            custom_config = '--psm 11 --oem 3'  # Page segmentation mode for sparse text
            ocr_data = pytesseract.image_to_data(
                screenshot, 
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )
            
            # Debug OCR results
            print("\nOCR Results:")
            found_words = []
            for i, word in enumerate(ocr_data['text']):
                if word.strip():  # Only log non-empty words
                    conf = float(ocr_data['conf'][i])
                    found_words.append(f"'{word}' (confidence: {conf:.1f}%)")
            print("Detected words:", ", ".join(found_words[:10]) + "..." if len(found_words) > 10 else ", ".join(found_words))
            
            best_match = None
            highest_confidence = 0
            search_text = text.lower()
            
            # Try different text matching strategies
            for i, word in enumerate(ocr_data['text']):
                if not word.strip():
                    continue
                
                word_lower = word.strip().lower()
                confidence = float(ocr_data['conf'][i])
                
                # Various matching strategies
                matched = False
                match_type = None
                
                # Exact match
                if search_text == word_lower:
                    matched = True
                    match_type = "exact"
                    confidence *= 1.2  # Boost confidence for exact matches
                
                # Contains match
                elif search_text in word_lower:
                    matched = True
                    match_type = "contains"
                    
                # Word is part of a longer text (e.g., "Save" in "Save As")
                elif word_lower in search_text:
                    matched = True
                    match_type = "partial"
                    confidence *= 0.8  # Reduce confidence for partial matches
                
                if matched and confidence > highest_confidence:
                    highest_confidence = confidence
                    x = ocr_data['left'][i] + ocr_data['width'][i] // 2
                    y = ocr_data['top'][i] + ocr_data['height'][i] // 2
                    best_match = (x, y, word, match_type, confidence)
            
            if best_match:
                x, y, matched_word, match_type, conf = best_match
                print(f"\nBest match: '{matched_word}' ({match_type} match, confidence: {conf:.1f}%)")
                print(f"Clicking at position: ({x}, {y})")
                
                # Move mouse first, then click
                pyautogui.moveTo(x, y, duration=0.2)
                pyautogui.click()
                
                self._speak(f"Clicking {matched_word}")
                return f"Clicked '{matched_word}' at ({x}, {y})"
            
            print("\nNo matching text found on screen")
            self._speak(f"Could not find {text} on screen")
            return "Text not found on screen"
            
        except Exception as e:
            error_msg = f"Click command failed: {str(e)}"
            print(error_msg)
            return error_msg

    async def _handle_type(self, text: str) -> str:
        """Handle type commands by using xdotool to type text."""
        try:
            # Capitalize first letter if original command was capitalized
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:]

            print(f"Typing text: '{text}'")
            subprocess.run(['xdotool', 'type', text], check=True)
            return f"Typed: '{text}'"
        except subprocess.CalledProcessError as e:
            error_msg = f"Type command failed: {str(e)}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected typing error: {str(e)}"
            print(error_msg)
            return error_msg

    async def _handle_computer(self, query: str) -> AsyncGenerator[str, None]:
        """Handle computer commands by processing with local LLM."""
        try:
            # Get highlighted text using xclip
            print("Getting highlighted text...")
            highlighted = subprocess.check_output(
                ['xclip', '-o', '-selection', 'primary'],
                stderr=subprocess.PIPE
            ).decode('utf-8').strip()
            
            if not highlighted:
                message = "No text is highlighted"
                self._speak(message)
                yield message
                return
                
            print(f"Processing query: '{query}' with context: '{highlighted[:100]}...'")
            
            # Handle define command specially
            if query.lower().startswith("define"):
                prompt = f"Define this word in a concise way: {highlighted}"
            else:
                prompt = f"Context: {highlighted}\nQuery: {query}"
            
            # Stream responses from Ollama
            async for response in self._stream_ollama(prompt):
                self._speak(response)
                yield response
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to get highlighted text: {str(e)}"
            print(error_msg)
            yield error_msg
        except Exception as e:
            error_msg = f"Computer command failed: {str(e)}"
            print(error_msg)
            yield error_msg

    async def _handle_read(self, text: str) -> str:
        """Handle read command by reading highlighted text aloud."""
        try:
            # Get highlighted text using xclip
            highlighted = subprocess.check_output(
                ['xclip', '-o', '-selection', 'primary'],
                stderr=subprocess.PIPE
            ).decode('utf-8').strip()
            
            if not highlighted:
                message = "No text is highlighted"
                self._speak(message)
                return message
            
            # Speak the highlighted text
            self._speak(highlighted)
            return f"Reading highlighted text: {highlighted[:50]}..."
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to get highlighted text: {str(e)}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Read command failed: {str(e)}"
            print(error_msg)
            return error_msg

    async def _stream_ollama(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream responses from Ollama."""
        try:
            print("Sending request to Ollama...")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": True
                },
                stream=True
            )
            
            if not response.ok:
                error_msg = f"Ollama request failed with status {response.status_code}"
                print(error_msg)
                yield error_msg
                return

            current_chunk = ""
            for line in response.iter_lines():
                if not line:
                    continue
                    
                chunk = json.loads(line)
                if 'response' in chunk:
                    current_chunk += chunk['response']
                    
                    # Yield complete sentences
                    if any(char in current_chunk for char in '.!?'):
                        yield current_chunk.strip()
                        current_chunk = ""
                        
            if current_chunk:  # Yield any remaining text
                yield current_chunk.strip()
                
        except requests.RequestException as e:
            error_msg = f"Failed to connect to Ollama: {str(e)}"
            print(error_msg)
            yield error_msg
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Ollama response: {str(e)}"
            print(error_msg)
            yield error_msg
        except Exception as e:
            error_msg = f"Unexpected error in Ollama processing: {str(e)}"
            print(error_msg)
            yield error_msg

    def _speak(self, text: str) -> None:
        """Speak text using espeak."""
        try:
            subprocess.run(['espeak', self.espeak_config, text], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Speech failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected speech error: {str(e)}")

    async def process_command(self, text: str) -> AsyncGenerator[str, None]:
        """Process a voice command and yield status messages."""
        command, args = self.parse_command(text)
        
        if not command:
            yield "No valid command found"
            return
            
        handler = self.commands.get(command)
        if not handler:
            yield f"Unknown command: {command}"
            return
            
        if handler.is_active:
            yield f"Command '{command}' is already running"
            return
            
        async with handler.running():
            try:
                result = handler.execute(args)
                if isinstance(result, AsyncGenerator):
                    async for message in result:
                        yield message
                else:
                    yield await result
            except Exception as e:
                error_msg = f"Command execution failed: {str(e)}"
                print(error_msg)
                yield error_msg
