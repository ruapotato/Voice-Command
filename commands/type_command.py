import subprocess
from .base import Command

class TypeCommand(Command):
    def __init__(self):
        super().__init__(
            name="type",
            aliases=["type in"],
            description="Type text using keyboard",
            execute=self._execute
        )

    async def _execute(self, text: str) -> str:
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
