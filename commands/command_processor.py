# commands/command_processor.py
import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple, Dict, List, Set
import inspect # Keep for process_command check

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
            if module_name in ['__init__', 'base']: continue
            full_module_name = f"commands.{module_name}"
            logger.debug(f"Attempting to import module: {full_module_name}")
            try:
                module = importlib.import_module(full_module_name)
                logger.debug(f"Successfully imported module: {full_module_name}")
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == full_module_name and issubclass(obj, Command) and obj is not Command:
                        try:
                            command_instance = obj()
                            if command_instance.name in self.commands: logger.warning(f"Duplicate command name '{command_instance.name}'. Overwriting.")
                            self.commands[command_instance.name] = command_instance
                            logger.debug(f"Registered command: '{command_instance.name}' from {full_module_name}")
                        except Exception as inst_e: logger.error(f"Failed to instantiate command {obj.__name__} from {full_module_name}: {inst_e}", exc_info=True)
            except ModuleNotFoundError: logger.error(f"Could not import module {full_module_name}. Skipping.", exc_info=True)
            except Exception as import_e: logger.error(f"Failed to import/process module {full_module_name}: {import_e}", exc_info=True)

    # get_all_command_names_aliases (remains same)
    def get_all_command_names_aliases(self) -> List[str]:
        all_triggers = set(self.commands.keys()); [all_triggers.update(cmd.aliases) for cmd in self.commands.values()]
        return sorted(list(all_triggers))

    # get_command_details (remains same)
    def get_command_details(self) -> List[Tuple[str, List[str], str]]:
         details = [ (cmd.name, cmd.aliases, getattr(cmd, 'description', 'No description.')) for cmd in sorted(self.commands.values(), key=lambda c: c.name) ]
         return details

    # parse_command (remains same)
    def parse_command(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        text = text.strip(); lower_text = text.lower()
        matched_command_name = None
        for cmd_name, command in self.commands.items():
             if lower_text == cmd_name: matched_command_name = cmd_name; break
             if lower_text in command.aliases: matched_command_name = cmd_name; break
        if matched_command_name: return matched_command_name, ""
        candidates = [];
        for cmd_name, command in self.commands.items(): candidates.append(cmd_name); candidates.extend(command.aliases)
        candidates.sort(key=len, reverse=True)
        for trigger in candidates:
             trigger_prefix = trigger + " "
             if lower_text.startswith(trigger_prefix):
                  cmd_name = None
                  if trigger in self.commands: cmd_name = trigger
                  else:
                       for name, command_obj in self.commands.items():
                            if trigger in command_obj.aliases: cmd_name = name; break
                  if cmd_name: args = text[len(trigger):].strip(); return cmd_name, args
        return None, None

    # --- CORRECTED process_command method ---
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
            execute_method = command.execute
            # --- Check type and execute/yield correctly ---
            if inspect.isasyncgenfunction(execute_method):
                # <<< FIX: Moved async for to next line >>>
                async for result_part in execute_method(args):
                    yield result_part
            elif inspect.iscoroutinefunction(execute_method):
                # <<< FIX: Moved await and yield to separate lines >>>
                result_message = await execute_method(args)
                yield result_message
            else:
                # Handle synchronous commands
                logger.warning(f"Command '{command_name}' execute method is synchronous.")
                # Consider loop.run_in_executor if blocking, but yield directly for now
                result_message = execute_method(args)
                yield result_message
        except Exception as e:
            error_msg = f"Command '{command_name}' execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg
