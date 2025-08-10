# commands/command_processor.py
import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple, Dict, List
from difflib import SequenceMatcher

from .base import Command

logger = logging.getLogger(__name__)

class CommandProcessor:
    def __init__(self):
        """Initialize the command processor by dynamically discovering commands."""
        self.commands: Dict[str, Command] = {}
        self.all_triggers_cache: List[str] = []
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
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == full_module_name and issubclass(obj, Command) and obj is not Command:
                        try:
                            command_instance = obj()
                            if command_instance.name in self.commands: logger.warning(f"Duplicate command name '{command_instance.name}'. Overwriting.")
                            self.commands[command_instance.name] = command_instance
                        except Exception as inst_e: logger.error(f"Failed to instantiate command {obj.__name__} from {full_module_name}: {inst_e}", exc_info=True)
            except Exception as import_e: logger.error(f"Failed to import/process module {full_module_name}: {import_e}", exc_info=True)
        self._cache_all_triggers()

    def _cache_all_triggers(self):
        """Builds and caches a sorted list of all command triggers."""
        triggers = set(self.commands.keys())
        for cmd in self.commands.values():
            triggers.update(cmd.aliases)
        self.all_triggers_cache = sorted(list(triggers), key=len, reverse=True)

    def get_command_details(self) -> List[Tuple[str, List[str], str]]:
        """Returns details for all registered commands for the help text."""
        details = [(cmd.name, cmd.aliases, getattr(cmd, 'description', 'No description.')) for cmd in sorted(self.commands.values(), key=lambda c: c.name)]
        return details

    def _get_command_name_for_trigger(self, trigger: str) -> Optional[str]:
        """Helper to find the main command name from a trigger (which could be an alias)."""
        if trigger in self.commands:
            return trigger
        for name, command_obj in self.commands.items():
            if trigger in command_obj.aliases:
                return name
        return None

    def parse_command(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses the command from the input text with robust prefix matching
        and fuzzy matching for single-word commands.
        """
        text_orig = text.strip()
        text_lower = text_orig.lower()

        # First, try robust prefix matching for all triggers.
        # This is better for commands that can take arguments.
        for trigger in self.all_triggers_cache:
            if text_lower.startswith(trigger):
                # If it's an exact match
                if len(text_lower) == len(trigger):
                    command_name = self._get_command_name_for_trigger(trigger)
                    return command_name, ""
                
                # If it's a prefix match (command with arguments)
                char_after_trigger = text_lower[len(trigger)]
                if char_after_trigger in ' ,.!?':
                    command_name = self._get_command_name_for_trigger(trigger)
                    if command_name:
                        args = text_orig[len(trigger):].lstrip(' ,.!?')
                        return command_name, args

        # If no prefix match, try fuzzy matching for single-word commands.
        # This helps with misspellings from voice-to-text.
        words = text_lower.split()
        if len(words) == 1:
            text_norm = words[0].rstrip('.,!?')
            # Find the best match among all triggers
            best_match_trigger = None
            highest_similarity = 0.85 # Minimum similarity threshold
            
            for trigger in self.all_triggers_cache:
                # Only fuzzy match against triggers that don't expect arguments usually
                # This is a heuristic: match against single-word triggers
                if " " not in trigger:
                    similarity = SequenceMatcher(None, text_norm, trigger).ratio()
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match_trigger = trigger
            
            if best_match_trigger:
                command_name = self._get_command_name_for_trigger(best_match_trigger)
                logger.debug(f"Fuzzy matched '{text_norm}' to '{best_match_trigger}' with similarity {highest_similarity:.2f}")
                return command_name, ""

        return None, None

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
            if inspect.isasyncgenfunction(execute_method):
                async for result_part in execute_method(args):
                    yield result_part
            elif inspect.iscoroutinefunction(execute_method):
                result_message = await execute_method(args)
                if result_message:
                    yield result_message
            else:
                logger.warning(f"Command '{command_name}' execute method is synchronous.")
                result_message = execute_method(args)
                if result_message:
                    yield result_message
        except Exception as e:
            error_msg = f"Command '{command_name}' execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg
