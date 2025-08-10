# commands/read_command.py
import subprocess
import logging
from typing import Optional
from .base import Command

# Add logger instance
logger = logging.getLogger(__name__)


class ReadCommand(Command):
    def __init__(self):
        super().__init__(
            name="read",
            aliases=["reed", "red", "three"], # Consider removing "three" if it's a misrecognition
            description="Read highlighted text aloud",
            execute=self._execute
        )
        self.espeak_config = []
        # Check if espeak exists on init
        try:
             subprocess.run(['which', 'espeak'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
             logger.error("'espeak' command not found. Read command will not work.")

    async def _execute(self, text: str) -> Optional[str]:
        """Handle read command by reading highlighted text aloud."""
        try:
            # Get highlighted text using xclip
            highlighted_process = subprocess.run(
                ['xclip', '-o', '-selection', 'primary'],
                capture_output=True, text=True, check=False, timeout=10
            )
            if highlighted_process.returncode != 0:
                 error_msg = "Failed to get highlighted text."
                 if "Error: Can't open display" in highlighted_process.stderr:
                      error_msg += " (Cannot open display)"
                 elif "Error: target STRING not available" in highlighted_process.stderr:
                      error_msg = "No text is highlighted (or not available as STRING)."
                 else:
                      error_msg += f" (xclip error: {highlighted_process.stderr.strip()})"
                 logger.warning(error_msg)
                 return error_msg

            highlighted = highlighted_process.stdout.strip()

            if not highlighted:
                message = "No text is highlighted."
                logger.info(message)
                self._speak(message)
                return message

            logger.info(f"Reading highlighted text (length: {len(highlighted)})...")
            self._speak(highlighted)

            return f"Finished reading highlighted text ({len(highlighted)} chars)."

        except FileNotFoundError:
             error_msg = "Error: 'xclip' command not found. Cannot read highlighted text."
             logger.error(error_msg)
             return error_msg
        except subprocess.TimeoutExpired:
             error_msg = "Error: 'xclip' command timed out."
             logger.error(error_msg)
             return error_msg
        except Exception as e:
            error_msg = f"Read command failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _speak(self, text: str) -> None:
        """Speak text using espeak with the command's config."""
        if not text:
            return
        try:
             command = ['espeak'] + self.espeak_config + [text]
             logger.debug(f"Executing internal speak: {' '.join(command)}")
             subprocess.run(['pkill', '-f', 'espeak'], check=False)
             subprocess.run(command, check=True)
        except FileNotFoundError:
             logger.error("Internal speak failed: 'espeak' command not found.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Internal speech failed (espeak error): {e}")
        except Exception as e:
            logger.error(f"Unexpected internal speech error: {str(e)}")
