import subprocess
import logging
from .base import Command

logger = logging.getLogger(__name__)

class StopCommand(Command):
    def __init__(self):
        super().__init__(
            name="stop",
            # Optional aliases:
            aliases=["cancel", "shutup", "silence"],
            description="Stops any active text-to-speech feedback (espeak).",
            execute=self._execute
            # State management isn't really needed for this command
        )

    async def _execute(self, args: str) -> str:
        """
        Executes the stop command by killing espeak processes.
        Args are ignored.
        """
        logger.info("Executing stop command...")
        try:
            # Use pkill to find and terminate espeak processes
            # '-f' matches against the full command line, which is safer
            # if espeak is potentially embedded in scripts.
            # We check the return code to see if any process was killed.
            result = subprocess.run(['pkill', '-f', 'espeak'], capture_output=True, check=False)

            if result.returncode == 0:
                logger.info("Successfully terminated espeak process(es).")
                return "Stopped active speech."
            elif result.returncode == 1:
                # pkill returns 1 if no processes matched
                logger.info("No espeak process found running.")
                return "No active speech found to stop."
            else:
                # Other errors (e.g., permission denied)
                error_msg = f"pkill command failed with code {result.returncode}: {result.stderr.decode('utf-8', errors='ignore').strip()}"
                logger.error(error_msg)
                return f"Error trying to stop speech: {error_msg}"

        except FileNotFoundError:
            logger.error("'pkill' command not found.")
            return "Error: 'pkill' command not found. Cannot stop speech."
        except Exception as e:
            error_msg = f"Unexpected error stopping speech: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
