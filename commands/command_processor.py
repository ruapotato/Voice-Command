from typing import AsyncGenerator, Optional, Tuple
from .base import Command
from .click_command import ClickCommand
from .type_command import TypeCommand
from .read_command import ReadCommand
from .computer_command import ComputerCommand

class CommandProcessor:
    def __init__(self, window=None):
        """Initialize the command processor with all available commands.
        
        Args:
            window: Optional reference to the main window for terminal integration
        """
        self.window = window
        computer_cmd = ComputerCommand()
        if window:
            computer_cmd.set_window(window)
            
        self.commands = {
            cmd.name: cmd for cmd in [
                ClickCommand(),
                TypeCommand(),
                ReadCommand(),
                computer_cmd
            ]
        }
        print("Command processor initialized with:", list(self.commands.keys()))

    def parse_command(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse the voice command into command type and arguments."""
        text = text.strip()
        lower_text = text.lower()

        for cmd_name, command in self.commands.items():
            # Check main command name
            if lower_text.startswith(cmd_name):
                args = text[len(cmd_name):].strip()
                return cmd_name, args
            
            # Check aliases
            for alias in command.aliases:
                if lower_text.startswith(alias):
                    args = text[len(alias):].strip()
                    return cmd_name, args

        return None, None

    async def process_command(self, text: str) -> AsyncGenerator[str, None]:
        """Process a voice command and yield status messages."""
        command_name, args = self.parse_command(text)
    
        if not command_name:
            yield "No valid command found"
            return
            
        command = self.commands.get(command_name)
        if not command:
            yield f"Unknown command: {command_name}"
            return
            
        if command.is_active:
            yield f"Command '{command_name}' is already running"
            return
            
        async with command.running():
            try:
                result = command.execute(args)
                if isinstance(result, AsyncGenerator):
                    async for message in result:
                        yield message
                else:
                    yield await result
            except Exception as e:
                error_msg = f"Command execution failed: {str(e)}"
                print(error_msg)
                yield error_msg

    def set_window(self, window):
        """Update window reference for command processor and commands.
        
        Args:
            window: Reference to the main window for terminal integration
        """
        self.window = window
        # Update window reference for commands that need it
        for cmd in self.commands.values():
            if hasattr(cmd, 'set_window'):
                cmd.set_window(window)
