import subprocess
import requests
import json
from typing import AsyncGenerator
from .base import Command

class ComputerCommand(Command):
    def __init__(self):
        super().__init__(
            name="computer",
            aliases=[],
            description="Ask questions about highlighted text",
            execute=self._execute
        )
        self.espeak_config = "-ven+f3 -k5 -s150"  # Speech configuration

    async def _execute(self, query: str) -> AsyncGenerator[str, None]:
        """Handle computer commands by processing with local LLM."""
        try:
            print("Getting highlighted text...")
            highlighted = subprocess.check_output(
                ['xclip', '-o', '-selection', 'primary'],
                stderr=subprocess.PIPE
            ).decode('utf-8').strip()
            
            if not highlighted:
                message = "No text is highlighted"
                yield message
                return
                
            print(f"Processing query: '{query}' with context: '{highlighted[:100]}...'")
            
            if query.lower().startswith("define"):
                prompt = f"Define this word in a concise way: {highlighted}"
            else:
                prompt = f"Context: {highlighted}\nQuery: {query}"
            
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
                    
                    if any(char in current_chunk for char in '.!?'):
                        yield current_chunk.strip()
                        current_chunk = ""
                        
            if current_chunk:
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
