# cli/completer.py
from prompt_toolkit.completion import Completer, Completion
from typing import Iterable, List, Set

# This list will be updated by main.py
ollama_models_for_completion: List[str] = ["mistral"]

class CLICompleter(Completer):
    def __init__(self, command_processor):
        """
        Initialize the completer.
        Args:
            command_processor: The initialized CommandProcessor instance.
        """
        self.command_processor = command_processor
        # <<< FIX: Define static_keywords BEFORE calling _update_command_triggers >>>
        self.static_keywords = sorted(["select", "help", "exit", "quit", "refresh_models"])
        self.select_options = ["model"]
        # Now call update, which uses self.static_keywords
        self._update_command_triggers() # Initial fetch

    def _update_command_triggers(self):
        """Updates the list of command names and aliases from the processor."""
        self.command_triggers: Set[str] = set()
        if self.command_processor:
            self.command_triggers.update(self.command_processor.commands.keys())
            for command in self.command_processor.commands.values():
                self.command_triggers.update(command.aliases)
        # Use the now defined self.static_keywords
        self.all_triggers = sorted(list(self.command_triggers.union(self.static_keywords)))
        # print(f"Completer updated triggers: {self.all_triggers}") # Debug

    # --- get_completions method remains the same ---
    def get_completions(self, document, complete_event):
        # (Previous implementation)
        text = document.text_before_cursor.lstrip()
        words = text.split()
        word_before_cursor = document.get_word_before_cursor(WORD=True)

        try:
            if not text or ' ' not in text: # Top Level Completion
                for trigger in self.all_triggers:
                    if trigger.startswith(word_before_cursor):
                        yield Completion(trigger, start_position=-len(word_before_cursor))
                return
            if len(words) >= 1: # Contextual Completion
                first_word = words[0]
                if first_word == "select": # 'select' command completion
                    if len(words) == 1 and text.endswith(' '):
                        for opt in self.select_options: yield Completion(opt, start_position=0)
                    elif len(words) == 2 and not text.endswith(' '): # Typing 'model'
                        if self.select_options[0].startswith(word_before_cursor): yield Completion(self.select_options[0], start_position=-len(word_before_cursor))
                    elif len(words) == 2 and words[1] == "model" and text.endswith(' '): # After 'select model '
                         for model in ollama_models_for_completion: yield Completion(model, start_position=0)
                    elif len(words) >= 3 and words[1] == "model": # Typing model name
                         for model in ollama_models_for_completion:
                             if model.startswith(word_before_cursor): yield Completion(model, start_position=-len(word_before_cursor))
                    return
        except Exception: pass # Avoid completer errors crashing app
