# commands/stop_command.py
import subprocess
import logging
from .base import Command
# <<< Import output functions >>>
from cli.output import schedule_print # Only need print for this command

logger = logging.getLogger(__name__)

class StopCommand(Command):
    def __init__(self):
        super().__init__(
            name="stop",
            aliases=["cancel", "shutup", "silence"],
            description="Stops any active text-to-speech feedback (espeak).",
            # <<< Reference the updated _execute >>>
            execute=self._execute
        )

    # <<< Updated signature and implementation >>>
    async def _execute(self, args: str) -> None:
        """
        Executes the stop command by killing espeak processes.
        Prints status to CLI, does not speak. Args are ignored.
        """
        logger.info("Executing stop command...")
        try:
            result = subprocess.run(['pkill', '-f', 'espeak'], capture_output=True, check=False)

            if result.returncode == 0:
                msg = "Stopped active speech."
                logger.info(msg)
                schedule_print("System", msg) # <<< Explicitly print
            elif result.returncode == 1:
                msg = "No active speech found to stop."
                logger.info(msg)
                schedule_print("System", msg) # <<< Explicitly print
            else:
                error_msg = f"pkill command failed with code {result.returncode}: {result.stderr.decode('utf-8', errors='ignore').strip()}"
                logger.error(error_msg)
                schedule_print("Error", f"Error trying to stop speech: {error_msg}") # <<< Explicitly print error

        except FileNotFoundError:
            error_msg = "Error: 'pkill' command not found. Cannot stop speech."
            logger.error(error_msg)
            schedule_print("Error", error_msg) # <<< Explicitly print error
        except Exception as e:
            error_msg = f"Unexpected error stopping speech: {str(e)}"
            logger.error(error_msg, exc_info=True)
            schedule_print("Error", error_msg) # <<< Explicitly print error
        # No return value needed now
