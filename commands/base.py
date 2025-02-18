# commands/base.py
from dataclasses import dataclass, field
from typing import Dict, Any, AsyncGenerator, Callable, Optional
from contextlib import asynccontextmanager

@dataclass(frozen=True)
class Command:
    """Base class for all commands."""
    name: str
    aliases: list[str]
    description: str
    execute: Callable
    state: Dict[str, bool] = field(default_factory=lambda: {'is_running': False})

    @property
    def is_active(self) -> bool:
        """Check if the command is currently running."""
        return self.state['is_running']
    
    @asynccontextmanager
    async def running(self):
        """Context manager for command execution state."""
        self.state['is_running'] = True
        try:
            yield
        finally:
            self.state['is_running'] = False
