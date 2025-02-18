import subprocess
from .base import Command

class ReadCommand(Command):
    def __init__(self):
        super().__init__(
            name="read",
            aliases=["reed", "red", "three"],
            description="Read highlighted text aloud",
            execute=self._execute
        )
        self.espeak_config = "-ven+f3 -k5 -s150"

    async def _execute(self, text: str) -> str:
        """Handle read command by reading highlighted text aloud."""
        try:
            highlighted = subprocess.check_output(
                ['xclip', '-o', '-selection', 'primary'],
                stderr=subprocess.PIPE
            ).decode('utf-8').strip()
            
            if not highlighted:
                message = "No text is highlighted"
                self._speak(message)
                return message
            
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

    def _speak(self, text: str) -> None:
        """Speak text using espeak."""
        try:
            subprocess.run(['espeak', self.espeak_config, text], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Speech failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected speech error: {str(e)}")
