# commands/base.py
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, Awaitable # Added Awaitable
from contextlib import asynccontextmanager

@dataclass(frozen=True) # Keep frozen for simplicity unless state needs mutation often
class Command:
    """Base class for all commands."""
    name: str
    aliases: list[str]
    description: str
    # <<< Updated signature: Now expects an async function that might not return anything significant >>>
    execute: Callable[[str], Awaitable[None]]
    # State might still be useful for complex, long-running commands, but less so now
    state: Dict[str, bool] = field(default_factory=lambda: {'is_running': False})

    @property
    def is_active(self) -> bool:
        """Check if the command is currently running (basic state check)."""
        return self.state['is_running']

    @asynccontextmanager
    async def running(self):
        """Context manager for command execution state (optional use)."""
        # This might be less necessary if commands are simpler now, but keep for potential use
        self.state['is_running'] = True
        try:
            yield
        finally:
            self.state['is_running'] = False
