# commands/read_command.py
import subprocess
import logging
from typing import Optional  # <<< Add this line
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
        # <<< CHANGE: Store config as a list >>>
        self.espeak_config = []
        # Check if espeak exists on init
        try:
             subprocess.run(['which', 'espeak'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
             logger.error("'espeak' command not found. Read command will not work.")
             # You might want to disable the command or handle this more gracefully

    async def _execute(self, text: str) -> Optional[str]: # Return Optional[str]
        """Handle read command by reading highlighted text aloud."""
        try:
            # Get highlighted text using xclip
            highlighted_process = subprocess.run(
                ['xclip', '-o', '-selection', 'primary'],
                capture_output=True, text=True, check=False, timeout=2 # Add timeout
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
                 # Speak the error message using the main speak function for consistency
                 # await speak(error_msg) # Decide if you want error spoken
                 return error_msg # Return the error message for printing

            highlighted = highlighted_process.stdout.strip()

            if not highlighted:
                message = "No text is highlighted."
                logger.info(message)
                # Speak this short message
                self._speak(message) # Use internal speak for this specific feedback
                return message # Return for printing

            # Speak the *actual* highlighted text
            logger.info(f"Reading highlighted text (length: {len(highlighted)})...")
            self._speak(highlighted)

            # <<< CHANGE: Return a simple confirmation, not for speaking >>>
            # Return None or a message that won't be picked up by the main speak logic
            return f"Finished reading highlighted text ({len(highlighted)} chars)."
            # Or simply: return None

        except FileNotFoundError:
             error_msg = "Error: 'xclip' command not found. Cannot read highlighted text."
             logger.error(error_msg)
             return error_msg # Return error for printing
        except subprocess.TimeoutExpired:
             error_msg = "Error: 'xclip' command timed out."
             logger.error(error_msg)
             return error_msg
        except Exception as e:
            error_msg = f"Read command failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg # Return error for printing

    def _speak(self, text: str) -> None:
        """Speak text using espeak with the command's config."""
        if not text:
            return
        try:
             # <<< CHANGE: Pass config correctly as part of the list >>>
             command = ['espeak'] + self.espeak_config + [text]
             logger.debug(f"Executing internal speak: {' '.join(command)}")
             # Kill any previous espeak instance before starting a new one from here
             subprocess.run(['pkill', '-f', 'espeak'], check=False)
             subprocess.run(command, check=True, timeout=30) # Increased timeout for longer text
        except FileNotFoundError:
             logger.error("Internal speak failed: 'espeak' command not found.")
        except subprocess.CalledProcessError as e:
            # Log error but don't necessarily crash the whole flow
            logger.error(f"Internal speech failed (espeak error): {e}")
        except subprocess.TimeoutExpired:
            logger.warning("Internal speech timed out.")
            # Ensure espeak is killed if it timed out
            subprocess.run(['pkill', '-f', 'espeak'], check=False)
        except Exception as e:
            logger.error(f"Unexpected internal speech error: {str(e)}")
