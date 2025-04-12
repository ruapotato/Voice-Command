# commands/command_processor.py
import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
# <<< ADDED typing imports >>>
from typing import AsyncGenerator, Optional, Tuple, Dict, List

from .base import Command

logger = logging.getLogger(__name__)

class CommandProcessor:
    def __init__(self):
        """Initialize the command processor by dynamically discovering commands."""
        self.commands: Dict[str, Command] = {}
        self._discover_commands()
        logger.info(f"Command processor dynamically loaded commands: {list(self.commands.keys())}")

    def _discover_commands(self):
        """Dynamically finds and registers command classes."""
        commands_package_path = Path(__file__).parent
        logger.debug(f"Discovering commands in: {commands_package_path}")

        for (_, module_name, _) in pkgutil.iter_modules([str(commands_package_path)]):
            # Avoid discovering __init__ itself or base class module
            if module_name in ['__init__', 'base']:
                continue
            full_module_name = f"commands.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == full_module_name and issubclass(obj, Command) and obj is not Command:
                        try:
                            command_instance = obj()
                            if command_instance.name in self.commands:
                                logger.warning(f"Duplicate command name '{command_instance.name}' found. Overwriting.")
                            self.commands[command_instance.name] = command_instance
                            logger.debug(f"Registered command: '{command_instance.name}' from {full_module_name}")
                        except Exception as inst_e:
                            logger.error(f"Failed to instantiate command {obj.__name__} from {full_module_name}: {inst_e}", exc_info=True)
            except ModuleNotFoundError:
                 # This can happen if a command file has an import error itself
                 logger.error(f"Could not import module {full_module_name}. Skipping. Check for errors within that file.", exc_info=True)
            except Exception as import_e:
                logger.error(f"Failed to process module {full_module_name}: {import_e}", exc_info=True)

    def get_all_command_names_aliases(self) -> List[str]:
        """Returns a sorted list of all command names and their aliases."""
        all_triggers = set(self.commands.keys())
        for command in self.commands.values():
            all_triggers.update(command.aliases)
        return sorted(list(all_triggers))

    def get_command_details(self) -> List[Tuple[str, List[str], str]]:
         """Returns a list of tuples: (name, aliases, description) for help text."""
         details = [
             # Ensure description exists, provide default if not
             (cmd.name, cmd.aliases, getattr(cmd, 'description', 'No description available.'))
             for cmd in sorted(self.commands.values(), key=lambda c: c.name)
         ]
         return details

    def parse_command(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse the voice/typed command into command name and arguments."""
        text = text.strip()
        lower_text = text.lower()

        # Check for exact match (command name or alias)
        matched_command_name = None
        for cmd_name, command in self.commands.items():
             if lower_text == cmd_name: matched_command_name = cmd_name; break
             if lower_text in command.aliases: matched_command_name = cmd_name; break # Map alias back
        if matched_command_name:
             # Check if this command typically requires args. If not, parse is complete.
             # Heuristic: If execute method signature shows args param beyond self, it might need them.
             # For now, assume exact match means no args intended by user here.
             return matched_command_name, ""

        # Check for prefix match (command name or alias followed by space)
        candidates = []
        for cmd_name, command in self.commands.items():
            candidates.append(cmd_name)
            candidates.extend(command.aliases)
        candidates.sort(key=len, reverse=True) # Match longer names first

        for trigger in candidates:
             trigger_prefix = trigger + " "
             if lower_text.startswith(trigger_prefix):
                  cmd_name = None
                  if trigger in self.commands: cmd_name = trigger
                  else: # Map alias back
                       for name, command_obj in self.commands.items():
                            if trigger in command_obj.aliases: cmd_name = name; break
                  if cmd_name:
                       args = text[len(trigger):].strip()
                       # Don't return if args are empty but command *needs* args? Hard to tell here.
                       # Assume prefix match implies args are intended.
                       return cmd_name, args

        return None, None # Let main.py handle fallback/general query

    async def process_command(self, text: str) -> AsyncGenerator[str, None]:
        """Process a command string and yield status messages."""
        command_name, args = self.parse_command(text)

        if not command_name:
            logger.warning(f"process_command called with unparseable text: {text}")
            yield f"Unknown command or query format: {text}"
            return

        command = self.commands.get(command_name)
        if not command:
            yield f"Internal error: Command '{command_name}' parsed but not found."
            return

        try:
            # Assuming command.execute is async def execute(self, args: str) -> str:
            # If it yields, we need to handle that differently
            # Based on current command impls, they return a single string.
            if inspect.isasyncgenfunction(command.execute):
                 # Handle async generator commands if any exist
                 async for result_part in command.execute(args):
                      yield result_part
            elif inspect.iscoroutinefunction(command.execute):
                 # Handle async function commands that return a single value
                 result_message = await command.execute(args)
                 yield result_message
            else:
                 # Handle synchronous commands (should run in executor ideally)
                 logger.warning(f"Command '{command_name}' execute method is synchronous.")
                 # For now, run directly, might block loop
                 result_message = command.execute(args)
                 yield result_message

        except Exception as e:
            error_msg = f"Command '{command_name}' execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg
