
import subprocess
import requests
import json
from typing import AsyncGenerator, Dict, List, Optional
import logging
import os
from pathlib import Path
from .base import Command

class ToolRegistry:
    """Registry for available system tools and their capabilities"""
    
    def __init__(self):
        self.apps: Dict[str, str] = {}  # name -> exec path
        self.active_windows: Dict[str, str] = {}  # window id -> name
        self.terminal_apps = ['konsole', 'gnome-terminal', 'xterm', 'terminator', 'alacritty', 'kitty']
        self.last_terminal_content = ""
        self.command_history = []
        self.update_installed_apps()
        self.update_active_windows()
    
    def update_installed_apps(self):
        """Update list of installed applications"""
        try:
            result = subprocess.run(
                ['find', '/usr/share/applications', '-name', '*.desktop'],
                capture_output=True, text=True
            )
            
            for desktop_file in result.stdout.splitlines():
                try:
                    with open(desktop_file, 'r') as f:
                        content = f.read()
                        name = None
                        exec_path = None
                        
                        for line in content.splitlines():
                            if line.startswith('Name='):
                                name = line.split('=')[1].lower()
                            elif line.startswith('Exec='):
                                exec_path = line.split('=')[1].split('%')[0].strip()
                                
                        if name and exec_path:
                            self.apps[name] = exec_path
                except:
                    continue
        except Exception as e:
            logging.error(f"Failed to update installed apps: {e}")

    def update_active_windows(self):
        """Update list of active windows using wmctrl"""
        try:
            result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True)
            self.active_windows.clear()
            
            for line in result.stdout.splitlines():
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    window_id, workspace, host, title = parts
                    self.active_windows[window_id] = title.lower()
        except Exception as e:
            logging.error(f"Failed to update active windows: {e}")

    def find_app(self, query: str) -> Optional[str]:
        """Find best matching installed application"""
        query = query.lower()
        
        # Special case for shell/terminal
        if query in ['shell', 'terminal']:
            for terminal in self.terminal_apps:
                exec_path = self.apps.get(terminal)
                if exec_path:
                    return exec_path
            return None
        
        for name, exec_path in self.apps.items():
            if query in name:
                return exec_path
        return None

    def find_window(self, query: str) -> Optional[str]:
        """Find best matching active window"""
        query = query.lower()
        for window_id, title in self.active_windows.items():
            if query in title:
                return window_id
        return None

    def add_command_history(self, command: str):
        """Add command to history"""
        self.command_history.append(command)
        if len(self.command_history) > 10:
            self.command_history.pop(0)

class ComputerCommand(Command):
    def __init__(self):
        super().__init__(
            name="computer",
            aliases=[],
            description="Execute various computer commands",
            execute=self._execute
        )
        self.espeak_config = "-ven+f3 -k5 -s150"
        self.tools = ToolRegistry()
        self.window = None 
        
        # Original query prompt for general questions
        self.query_prompt = """Context: {highlighted}
Query: {query}

Analyze the highlighted text and answer the query. Keep responses clear and concise.
If the query isn't directly related to the highlighted text, mention that.
If no text is highlighted, mention that we need highlighted text to answer the query."""

        # Shell command system prompt
        self.shell_prompt = """You are a desktop command assistant that outputs ONLY a single bash command.

Rules:
1. Core Response Types:
   - General queries: Use espeak for direct responses
   - File/system commands: Use appropriate bash commands
   - ALWAYS use espeak for feedback after commands

2. Command Usage:
   - File operations: ls, pwd, cat (only for existing files)
   - System info: ps, top, df
   - Navigation: cd (with valid paths)
   - ALWAYS include user feedback with espeak

3. Response Guidelines:
   - Keep responses relevant and contextual
   - Don't create fake files or paths
   - Include espeak feedback for ALL commands
   - Stay on current topic

4. Command Structure:
   - Output exactly ONE command
   - Always combine command with feedback
   - Use && to chain command with espeak
   - Remove any backticks or dangerous operations

Examples:
User: "check disk space"
Assistant: df -h && espeak "Here is your disk usage information"

User: "show current directory"
Assistant: pwd && espeak "This is your current location"

User: "list files"
Assistant: ls && espeak "Listing files in current directory"

Current terminal state:
{terminal_content}

Previous commands:
{command_history}

User request: {query}"""

        # Tool-specific prompts
        self.tool_prompts = {
            'open': """You are an application launcher assistant. Your only task is to help users open applications.

Available tool:
<open>app_name</open> - Launch an application

Rules:
1. You can ONLY use the <open> tool
2. Match the user's request against the list of installed applications
3. Choose the best matching application
4. For "shell" or "terminal" requests, find any installed terminal emulator
5. If no match is found, explain that the app isn't installed

Examples:
User: "open firefox" → <open>firefox</open>
User: "launch text editor" → <open>kate</open>
User: "open shell" → <open>terminal</open>

Installed applications:
{apps}

User request: {query}""",

            'window': """You are a window management assistant. Your task is to help users manage their open windows.

Available tools:
<goto>window_name</goto> - Focus a window
<close>window_name</close> - Close a window

Rules:
1. You can ONLY use the <goto> or <close> tools
2. Match the user's request against the list of active windows
3. For "close" commands, always use <close> even if the verb is "closed"
4. For "goto" commands, match both "goto" and "go to" variants
5. If no match is found, explain that the window isn't open

Examples:
User: "go to browser" → <goto>firefox</goto>
User: "close terminal" → <close>konsole</close>
User: "closed editor" → <close>kate</close>
User: "goto editor" → <goto>kate</goto>

Active windows:
{windows}

User request: {query}"""
        }

    async def _execute(self, text: str) -> AsyncGenerator[str, None]:
        """Process and execute computer commands"""
        try:
            # Determine if this is a tool command or a general query
            tool_type = self._determine_tool_type(text)
            
            if tool_type == 'query':
                # Handle as original computer command with highlighted text
                async for response in self._handle_text_query(text):
                    yield response
                return
                
            if tool_type == 'shell':
                # Handle shell commands
                async for response in self._handle_shell_command(text):
                    yield response
                return
            
            # Handle tool-based commands
            response = await self._get_llm_response(text, tool_type)
            if not response:
                yield "Failed to process command"
                return
                
            tool, params = self._parse_tool(response)
            if not tool:
                yield f"Unsupported command: {text}"
                return
                
            result = await self._execute_tool(tool, params)
            yield result
            
        except Exception as e:
            error_msg = f"Command failed: {str(e)}"
            logging.error(error_msg)
            yield error_msg

    def _determine_tool_type(self, query: str) -> str:
        """Determine which tool prompt to use based on the query"""
        query = query.lower()
        words = query.split()
        
        # First check for explicit shell command
        if query.startswith('shell '):
            return 'shell'
        
        # Then check other tool patterns
        tool_patterns = {
            'window': [
                'goto', 'go to', 'switch to', 'focus', 
                'close', 'closed', 'quit', 'exit'
            ],
            'open': ['open', 'launch', 'start', 'run'],
        }
        
        # Check for explicit tool commands
        for tool, patterns in tool_patterns.items():
            if any(pattern in query for pattern in patterns):
                return tool
        
        # If no explicit tool command is found, treat as a query
        return 'query'

    def set_window(self, window):
        """Set reference to main window for terminal integration."""
        self.window = window


    async def _handle_shell_command(self, query: str) -> AsyncGenerator[str, None]:
        """Handle shell commands using the terminal window and LLM"""
        try:
            # Extract the actual command/query after "shell"
            command_query = query.lower().replace('shell', '', 1).strip()
            
            if not command_query:
                yield "No command specified"
                return
                
            # Format prompt with terminal context
            history_context = "\n".join(
                f"Previous command: {cmd}" 
                for cmd in self.tools.command_history[-3:]
            )
            
            prompt = self.shell_prompt.format(
                terminal_content=self.tools.last_terminal_content,
                command_history=history_context,
                query=command_query
            )
            
            # Get command from LLM
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if not response.ok:
                yield f"Failed to get command: {response.status_code}"
                return
                
            # Extract the command from LLM response
            command = response.json()['response'].strip().split('\n')[0]
            
            # Execute in terminal window
            if self.window:
                success = self.window.execute_shell_command(command)
                if success:
                    self.tools.add_command_history(command)
                    yield f"Executing: {command}"
                else:
                    yield f"Failed to execute: {command}"
            else:
                yield "Terminal window not available"
                
        except Exception as e:
            error_msg = f"Shell command failed: {str(e)}"
            logging.error(error_msg)
            yield error_msg

    async def _handle_text_query(self, query: str) -> AsyncGenerator[str, None]:
        """Handle general queries using highlighted text as context"""
        try:
            print("Getting highlighted text...")
            highlighted = subprocess.check_output(
                ['xclip', '-o', '-selection', 'primary'],
                stderr=subprocess.PIPE
            ).decode('utf-8').strip()
            
            if not highlighted:
                message = "No text is highlighted"
                yield message
                self._speak(message)
                return
                
            print(f"Processing query: '{query}' with context: '{highlighted[:100]}...'")
            
            prompt = self.query_prompt.format(
                highlighted=highlighted,
                query=query
            )
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": True
                }
            )
            
            if not response.ok:
                error_msg = f"Query failed with status {response.status_code}"
                print(error_msg)
                yield error_msg
                return

            current_chunk = ""
            for line in response.iter_lines():
                if not line:
                    continue
                    
                chunk = json.loads(line)
                if 'response' in chunk:
                    current_chunk += chunk['response']
                    
                    if any(char in current_chunk for char in '.!?'):
                        response_text = current_chunk.strip()
                        self._speak(response_text)
                        yield response_text
                        current_chunk = ""
                        
            if current_chunk:
                response_text = current_chunk.strip()
                self._speak(response_text)
                yield response_text
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to get highlighted text: {str(e)}"
            print(error_msg)
            yield error_msg
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            print(error_msg)
            yield error_msg

    async def _get_llm_response(self, query: str, tool_type: str) -> Optional[str]:
        """Get tool selection from local LLM using specific prompt"""
        try:
            # Get appropriate prompt template
            prompt_template = self.tool_prompts.get(tool_type)
            if not prompt_template:
                return None
                
            # Format prompt with context
            prompt = prompt_template.format(
                apps="\n".join(self.tools.apps.keys()),
                windows="\n".join(self.tools.active_windows.values()),
                query=query
            )
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.ok:
                return response.json()['response'].strip()
            return None
            
        except Exception as e:
            logging.error(f"LLM error: {e}")
            return None

    def _parse_tool(self, response: str) -> tuple[Optional[str], Optional[str]]:
        """Parse tool type and parameters from LLM response"""
        try:
            for tool in ['open', 'goto', 'close', 'shell']:
                start_tag = f"<{tool}>"
                end_tag = f"</{tool}>"
                
                if start_tag in response and end_tag in response:
                    start_idx = response.index(start_tag) + len(start_tag)
                    end_idx = response.index(end_tag)
                    params = response[start_idx:end_idx].strip()
                    return tool, params
                    
            return None, None
            
        except Exception as e:
            logging.error(f"Failed to parse tool: {e}")
            return None, None

    async def _execute_tool(self, tool_type: str, params: str) -> str:
        """Execute the selected tool"""
        try:
            if tool_type == 'open':
                exec_path = self.tools.find_app(params)
                if exec_path:
                    subprocess.Popen(exec_path.split())
                    return f"Launched {params}"
                return f"Application not found: {params}"
                
            elif tool_type == 'goto':
                window_id = self.tools.find_window(params)
                if window_id:
                    subprocess.run(['wmctrl', '-i', '-a', window_id])
                    return f"Focused window: {params}"
                return f"Window not found: {params}"
                
            elif tool_type == 'close':
                window_id = self.tools.find_window(params)
                if window_id:
                    subprocess.run(['wmctrl', '-i', '-c', window_id])
                    return f"Closed window: {params}"
                return f"Window not found: {params}"
                
            elif tool_type == 'shell':
                # Use terminal window for shell commands
                if self.window:
                    success = self.window.execute_shell_command(params)
                    return f"Executed in terminal: {params}" if success else f"Failed to execute: {params}"
                return "Terminal window not available"
            
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    def _speak(self, text: str) -> None:
        """Speak text using espeak"""
        try:
            subprocess.run(['espeak', self.espeak_config, text], check=True)
        except Exception as e:
            logging.error(f"Speech failed: {e}")
