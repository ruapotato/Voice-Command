import subprocess
import requests # Keep for sync ToolRegistry updates if needed
import json
from typing import AsyncGenerator, Dict, List, Optional
import logging
import os
from pathlib import Path
from .base import Command
import aiohttp # Use aiohttp for async requests
import asyncio # Needed for asyncio.TimeoutError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ToolRegistry class remains the same as original...
class ToolRegistry:
    """Registry for available system tools and their capabilities"""
    # ... (Previous implementation from corrected version) ...
    def __init__(self):
        self.apps: Dict[str, str] = {}  # name -> exec path
        self.active_windows: Dict[str, str] = {}  # window id -> name
        self.terminal_apps = ['konsole', 'gnome-terminal', 'xterm', 'terminator', 'alacritty', 'kitty']
        self.last_terminal_content = ""
        self.command_history = []
        # Run updates synchronously in init for now
        self.update_installed_apps()
        self.update_active_windows()

    def update_installed_apps(self):
        """Update list of installed applications"""
        try:
            result = subprocess.run(
                ['find', '/usr/share/applications', '-name', '*.desktop'],
                capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                 logger.warning(f"Finding .desktop files failed: {result.stderr}")
                 return
            # ... rest of parsing logic ...
            # (Make sure this parsing is robust)
            for desktop_file in result.stdout.splitlines():
                try:
                    with open(desktop_file, 'r') as f:
                        content = f.read()
                        name = None
                        exec_path = None
                        nodisplay = False

                        for line in content.splitlines():
                             if line.startswith('Name='):
                                 name = line.split('=', 1)[1].lower().strip()
                             elif line.startswith('Exec='):
                                 exec_path = line.split('=', 1)[1].split('%')[0].strip()
                             elif line.startswith('NoDisplay=true'):
                                 nodisplay = True
                        if name and exec_path and not nodisplay:
                             if '/' in exec_path or subprocess.run(['which', exec_path.split()[0]], capture_output=True, check=False).returncode == 0:
                                self.apps[name] = exec_path

                except Exception as file_e:
                     logger.warning(f"Could not parse desktop file {desktop_file}: {file_e}")
                     continue

        except Exception as e:
            logging.error(f"Failed to update installed apps: {e}")


    def update_active_windows(self):
        """Update list of active windows using wmctrl"""
        try:
            result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                 logger.warning(f"wmctrl command failed: {result.stderr}. Window commands disabled.")
                 self.active_windows.clear()
                 return

            self.active_windows.clear()
            for line in result.stdout.splitlines():
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    window_id, workspace, host, title = parts
                    self.active_windows[window_id] = title.lower() # Store lowercase title
        except FileNotFoundError:
             logger.error("wmctrl not found. Window management commands unavailable.")
             self.active_windows.clear()
        except Exception as e:
            logging.error(f"Failed to update active windows: {e}")
            self.active_windows.clear()

    # ... find_app, find_window, add_command_history methods ...
    def find_app(self, query: str) -> Optional[str]:
        """Find best matching installed application"""
        query = query.lower()
        if query in ['shell', 'terminal', 'command prompt', 'cmd']:
            for terminal in self.terminal_apps:
                if terminal in self.apps: return self.apps[terminal]
                if terminal_exec := self.apps.get(terminal.split()[0]): return terminal_exec
            logger.warning("Could not find a known terminal application.")
            return None
        if query in self.apps: return self.apps[query]
        for name, exec_path in self.apps.items():
            if query in name:
                logger.debug(f"Partial match for '{query}': Found '{name}' -> {exec_path}")
                return exec_path
        logger.debug(f"No application found matching query: '{query}'")
        return None

    def find_window(self, query: str) -> Optional[str]:
        """Find best matching active window"""
        query = query.lower()
        if not self.active_windows:
             logger.warning("Active window list is empty or failed to update.")
             return None
        # Check exact lowercase match first
        for window_id, title in self.active_windows.items():
            if query == title:
                return window_id
        # Check partial match (contains query)
        best_match_id = None
        best_match_score = 0 # Higher score for better match (e.g., startswith, length)
        for window_id, title in self.active_windows.items():
            if query in title:
                 score = 100 - len(title) # Simple score: prefer shorter matching titles
                 if title.startswith(query):
                      score += 50 # Boost score if title starts with query
                 if score > best_match_score:
                      best_match_id = window_id
                      best_match_score = score
        if best_match_id:
            logger.debug(f"Window match for '{query}': Found '{self.active_windows[best_match_id]}' (ID: {best_match_id}, Score: {best_match_score})")
            return best_match_id
        logger.debug(f"No active window found matching query: '{query}'")
        return None

    def add_command_history(self, command: str):
        """Add command to history"""
        self.command_history.append(command)
        if len(self.command_history) > 10: self.command_history.pop(0)


# --- ComputerCommand Class ---
class ComputerCommand(Command):
    def __init__(self):
        super().__init__(
            name="computer",
            aliases=[],
            description="Execute various computer commands using LLM",
            execute=self._execute
        )
        self.espeak_config = "-ven+f3 -k5 -s150"
        self.tools = ToolRegistry()
        self.llm_model = "mistral"
        self.ollama_base_url = "http://localhost:11434"
        # --- Prompts --- (Keep previous prompts)
        self.query_prompt = """Context of highlighted text: "{highlighted}"
Now for the User Query: "{query}"

Analyze the highlighted text and answer the query. Keep responses clear and concise.
If the query isn't directly related to the highlighted text, just answer the qestion."""
        self.shell_prompt = """You are a desktop command assistant that outputs ONLY a single BASH command suitable for execution via subprocess.run.

Rules:
1. Task Handling:
   - If the request asks for information obtainable via a bash command (e.g., disk space, list files, current directory), output the command.
   - If the request is a general question or cannot be answered by a simple command, respond conversationally using ONLY plain text (no command output). Start conversational responses with 'ANSWER:'.
   - Provide ONLY the command itself (e.g., `ls -l`) or the conversational answer (e.g., `ANSWER: I cannot perform that action.`). Do NOT add explanations before the command or ANSWER:.
2. Safety:
   - AVOID destructive commands (rm, mv without care, mkfs, etc.). Prefer read-only commands (ls, pwd, df, ps, top, cat, head, tail, grep, find).
   - Do NOT create files or directories unless specifically asked and safe (e.g., `mkdir temp_dir`).
   - Validate paths if possible, but prefer commands that work relative to the current directory (`pwd`, `ls`).
   - Do NOT include `sudo` or attempt privilege escalation.
   - Do NOT include `&& espeak ...` or any other feedback mechanism in the command string itself. Feedback is handled separately.
3. Formatting:
   - Output exactly ONE line containing either the bash command or the `ANSWER:` prefixed conversational response.
   - Remove any markdown formatting like backticks (`).

Examples:
User: "check disk space"
Assistant: df -h

User: "show current directory"
Assistant: pwd

User: "list files"
Assistant: ls -lah

User: "what is the capital of france"
Assistant: ANSWER: The capital of France is Paris.

User: "delete all my files"
Assistant: ANSWER: I cannot perform destructive actions like deleting all files.

Current state (informational only, do not rely on for paths):
Working Directory (approximated): {working_dir}
Previous Commands (for context):
{command_history}

User request: {query}
Assistant:"""
        self.tool_prompts = {
            'open': """You are an application launcher assistant. Your task is to help users open applications by generating an <open>tool_name</open> tag.

Available tool:
<open>app_name</open> - Request to launch an application by its name.

Rules:
1. You can ONLY use the <open> tool format.
2. Analyze the user request and identify the application they want to open.
3. Match the user's request against the list of installed applications provided below.
4. Choose the best matching application name from the list. Use lowercase.
5. For "shell" or "terminal" requests, use the name of an installed terminal emulator from the list (e.g., <open>konsole</open>).
6. If no suitable match is found in the list, respond with: <open>NOT_FOUND</open>
7. Output ONLY the <open> tag. Do not add any other text.

Examples:
User: "open firefox" -> <open>firefox</open>
User: "launch the text editor" (kate is installed) -> <open>kate</open>
User: "start my browser" (firefox is installed) -> <open>firefox</open>
User: "open shell" (konsole is installed) -> <open>konsole</open>
User: "run gimp" (gimp not installed) -> <open>NOT_FOUND</open>

Installed applications:
{apps}

User request: {query}
Assistant:""",
            'window': """You are a window management assistant. Your task is to help users manage open windows by generating a <goto> or <close> tag.

Available tools:
<goto>window_query</goto> - Request to focus (switch to) a window based on its title or app name.
<close>window_query</close> - Request to close a window based on its title or app name.

Rules:
1. You can ONLY use the <goto> or <close> tool format.
2. Analyze the user request and identify the action (goto/close) and the target window.
3. Match the target window against the list of active window titles provided below. Use a keyword from the title or application name. Use lowercase.
4. For "close" commands, use the <close> tag.
5. For "focus", "switch to", or "go to" commands, use the <goto> tag.
6. If no suitable window is found in the list, respond with: <goto>NOT_FOUND</goto> or <close>NOT_FOUND</close> as appropriate.
7. Output ONLY the single tag. Do not add any other text.

Examples:
User: "go to browser" (firefox is open) -> <goto>firefox</goto>
User: "switch to the document" (document - libreoffice writer is open) -> <goto>document</goto>
User: "close terminal" (konsole is open) -> <close>konsole</close>
User: "close the editor" (kate is open) -> <close>kate</close>
User: "focus gimp" (gimp not open) -> <goto>NOT_FOUND</goto>

Active windows (lowercase titles):
{windows}

User request: {query}
Assistant:"""
        }


    async def get_available_models(self) -> Optional[List[str]]:
        """Asynchronously fetches available models from the Ollama API."""
        # ... (Implementation from previous correction) ...
        url = f"{self.ollama_base_url}/api/tags"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models_data = data.get("models", [])
                        model_names = sorted([m.get("name") for m in models_data if m.get("name")])
                        logger.info(f"Fetched Ollama models: {model_names}")
                        return model_names if model_names else None # Return None if empty list
                    else:
                        logger.error(f"Ollama API model request failed: {response.status} - {await response.text()}")
                        return None
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
             logger.error(f"Cannot connect to Ollama at {url}: {e}")
             return None
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}", exc_info=True)
            return None

    def set_llm_model(self, model_name):
        self.llm_model = model_name
        logger.info(f"ComputerCommand LLM model set to: {model_name}")

    async def _execute(self, text: str) -> AsyncGenerator[str, None]:
        logger.debug(f"ComputerCommand executing with text: '{text}'")
        try:
            tool_type = self._determine_tool_type(text)
            logger.debug(f"Determined tool type: {tool_type}")

            if tool_type == 'query':
                async for response in self._handle_text_query(text): yield response
            elif tool_type == 'shell':
                 command_query = text.lower().replace('shell', '', 1).strip()
                 async for response in self._handle_shell_command(command_query): yield response
            else: # 'open' or 'window'
                 llm_response_tag = "[No response]" # Default
                 # Use the generator and get the first (and only) result for non-streaming
                 async for resp in self._get_llm_tool_response(text, tool_type):
                     llm_response_tag = resp
                     break # Only need the first yielded item

                 if llm_response_tag.startswith("[Error:") or llm_response_tag == "[No response]":
                    yield f"Failed to get command instruction from LLM for '{tool_type}': {llm_response_tag}"
                    return

                 tool_name, params = self._parse_tool(llm_response_tag)
                 logger.debug(f"Parsed tool: {tool_name}, Params: {params}")

                 if not tool_name:
                    yield f"LLM did not provide a valid tool command: {llm_response_tag}"
                    return
                 if params is not None and params.upper() == "NOT_FOUND":
                     yield f"Could not find the requested {tool_type} target '{text}'."
                     return

                 result = await self._execute_tool(tool_name, params if params else "") # Pass empty string if params is None
                 yield result
        except Exception as e:
            error_msg = f"ComputerCommand execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg

    def _determine_tool_type(self, query: str) -> str:
        query_lower = query.lower().strip()
        # Check for prefixes first
        if query_lower.startswith('shell'): return 'shell'
        if query_lower.startswith('open'): return 'open'
        if query_lower.startswith('launch'): return 'open'
        if query_lower.startswith('start'): return 'open'
        if query_lower.startswith('run'): return 'open'
        if query_lower.startswith('goto'): return 'window'
        if query_lower.startswith('go to'): return 'window'
        if query_lower.startswith('switch to'): return 'window'
        if query_lower.startswith('focus'): return 'window'
        if query_lower.startswith('close'): return 'window'
        if query_lower.startswith('quit'): return 'window' # Could be ambiguous
        if query_lower.startswith('exit'): return 'window' # Could be ambiguous

        # If no prefix matches, assume 'query' (needs highlighted text)
        # General queries are handled by main.py's fallback mechanism
        logger.debug(f"Query '{query}' classified as 'query' type (needs 'computer' prefix and highlighted text).")
        return 'query'

    async def _handle_shell_command(self, command_query: str) -> AsyncGenerator[str, None]:
        # ... (Implementation from previous correction - ensure it uses async _ollama_generate correctly) ...
        if not command_query: yield "No shell command requested."; return
        logger.debug("Handling shell command request...")
        history_context = "\n".join(self.tools.command_history[-3:])
        current_dir = os.getcwd()
        prompt = self.shell_prompt.format(working_dir=current_dir, command_history=history_context, query=command_query)

        llm_output_line = "[No response]"
        async for resp in self._ollama_generate(prompt, stream=False): # Get single result
            llm_output_line = resp
            break

        if llm_output_line.startswith("[Error:") or llm_output_line == "[No response]":
             yield f"LLM failed for shell command: {llm_output_line}"
             return

        llm_output_line = llm_output_line.strip()
        logger.debug(f"LLM output for shell: '{llm_output_line}'")

        if llm_output_line.startswith("ANSWER:"):
             yield llm_output_line[len("ANSWER:"):].strip()
        elif not llm_output_line:
             yield "LLM returned empty response."
        else:
             command_to_run = llm_output_line
             yield f"Suggested command: `{command_to_run}`"
             # *** Add confirmation step here? ***
             # confirmation = await prompt_toolkit_confirm("Execute this command? (y/n)") -> Needs integration with main loop
             # if confirmation:
             yield f"Executing..."
             try:
                 # Run in executor to avoid blocking main loop
                 loop = asyncio.get_running_loop()
                 result = await loop.run_in_executor(
                     None, # Use default executor
                     lambda: subprocess.run(
                         command_to_run, shell=True, capture_output=True,
                         text=True, check=False, timeout=15
                     )
                 )
                 output = f"Return Code: {result.returncode}\n"
                 if result.stdout: output += f"Output:\n{result.stdout.strip()}\n"
                 if result.stderr: output += f"Error Output:\n{result.stderr.strip()}"
                 yield output.strip()
                 self.tools.add_command_history(command_to_run)
             except subprocess.TimeoutExpired: yield f"Command timed out: `{command_to_run}`"
             except Exception as exec_e: yield f"Failed to execute command `{command_to_run}`: {exec_e}"
             # else: yield "Execution cancelled."


    async def _handle_text_query(self, query: str) -> AsyncGenerator[str, None]:
        # ... (Implementation from previous correction - uses async _ollama_generate stream=True) ...
        try:
            logger.debug("Getting highlighted text for query context...")
            result = subprocess.run(['xclip', '-o', '-selection', 'primary'], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                 if "Error: target PRIMARY not available" in result.stderr or "Error: Can't open display" in result.stderr:
                      yield "Could not get highlighted text (display or selection error)."
                 else: yield "No text highlighted to use as context."
                 return
            highlighted = result.stdout.strip()
            if not highlighted: yield "No text is highlighted."; return

            logger.debug(f"Processing query: '{query}' with context: '{highlighted[:100]}...'")
            prompt = self.query_prompt.format(highlighted=highlighted, query=query)
            # Stream the response
            async for chunk_text in self._ollama_generate(prompt, stream=True):
                yield chunk_text
        except FileNotFoundError: yield "Error: 'xclip' command not found."; logger.error("'xclip' not found.")
        except Exception as e: error_msg = f"Query processing failed: {str(e)}"; logger.error(error_msg, exc_info=True); yield error_msg


    async def _get_llm_tool_response(self, query: str, tool_type: str) -> AsyncGenerator[str, None]:
        """Gets tool tag from LLM. Now an async generator yielding one result."""
        prompt_template = self.tool_prompts.get(tool_type)
        if not prompt_template:
            logger.error(f"No prompt template found for tool type: {tool_type}")
            yield f"[Error: No prompt for tool {tool_type}]"
            return

        self.tools.update_active_windows() # Update context before prompt
        apps_list = "\n".join(self.tools.apps.keys())
        windows_list = "\n".join(self.tools.active_windows.values())

        prompt = prompt_template.format(
            apps=apps_list if apps_list else "No applications found.",
            windows=windows_list if windows_list else "No active windows found.",
            query=query
        )
        logger.debug(f"Prompt for {tool_type} tool:\n{prompt}")

        # Use the generator, will yield one result or error
        async for response_text in self._ollama_generate(prompt, stream=False):
             logger.debug(f"LLM response for {tool_type} tool: {response_text}")
             yield response_text


    def _parse_tool(self, response: str) -> tuple[Optional[str], Optional[str]]:
        # ... (Implementation from previous correction - no changes needed here) ...
        response = response.strip()
        for tool in ['open', 'goto', 'close', 'shell']:
            start_tag = f"<{tool}>"
            end_tag = f"</{tool}>"
            if response.startswith(start_tag) and response.endswith(end_tag):
                start_idx = len(start_tag)
                end_idx = len(response) - len(end_tag)
                params = response[start_idx:end_idx].strip()
                if params: return tool, params
                else: logger.warning(f"Parsed empty params for tool {tool}"); return None, None
        logger.debug(f"Could not parse valid tool tag from: {response}")
        return None, None

    async def _execute_tool(self, tool_type: str, params: str) -> str:
        # ... (Implementation from previous correction - Popen/run can stay sync for now) ...
        logger.info(f"Executing tool '{tool_type}' with params '{params}'")
        try:
            if tool_type == 'open':
                exec_path = self.tools.find_app(params)
                if exec_path:
                    try: subprocess.Popen(exec_path.split(), start_new_session=True); return f"Attempting launch: '{params}'."
                    except Exception as e: return f"Error launching '{params}': {e}"
                else: return f"Application matching '{params}' not found."
            elif tool_type == 'goto':
                self.tools.update_active_windows()
                window_id = self.tools.find_window(params)
                if window_id:
                    try: subprocess.run(['wmctrl', '-i', '-a', window_id], check=True, timeout=2); return f"Focused: '{params}'."
                    except Exception as e: return f"Error focusing '{params}': {e}"
                else: return f"Window matching '{params}' not found."
            elif tool_type == 'close':
                self.tools.update_active_windows()
                window_id = self.tools.find_window(params)
                if window_id:
                    try: subprocess.run(['wmctrl', '-i', '-c', window_id], check=True, timeout=2); return f"Closed: '{params}'."
                    except Exception as e: return f"Error closing '{params}': {e}"
                else: return f"Window matching '{params}' not found."
            else: return f"Unknown tool type: {tool_type}"
        except Exception as e: logger.error(f"Tool execution failed: {e}", exc_info=True); return f"Tool execution failed: {str(e)}"

    def _speak(self, text: str) -> None:
        # ... (Implementation from previous correction - sync is likely fine) ...
        logger.debug(f"Speaking: {text[:50]}...")
        try: subprocess.run(['espeak', self.espeak_config, text], check=True, timeout=10)
        except FileNotFoundError: logger.error("espeak command not found.")
        except subprocess.TimeoutExpired: logger.warning(f"espeak command timed out.")
        except Exception as e: logger.error(f"Speech failed: {e}")


    # --- CORRECTED Helper for Ollama API calls ---
    async def _ollama_generate(self, prompt: str, stream: bool) -> AsyncGenerator[str, None]:
        """
        Internal helper to call Ollama generate API. Always returns an async generator.
        For non-streaming calls, yields the single result/error then stops.
        """
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": stream
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error {response.status}: {error_text}")
                        yield f"[Error: Ollama API failed ({response.status})]"
                        return # Stop generation

                    if stream:
                         buffer = ""
                         async for line in response.content:
                             if line:
                                 try:
                                     decoded_line = line.decode('utf-8')
                                     data = json.loads(decoded_line)
                                     chunk_text = data.get('response', '')
                                     if chunk_text:
                                        buffer += chunk_text
                                        # Yield completed sentences/fragments
                                        while True:
                                             try:
                                                  split_idx = min(idx for idx in (buffer.find('.'), buffer.find('!'), buffer.find('?'), buffer.find('\n')) if idx != -1)
                                                  yield buffer[:split_idx+1]
                                                  buffer = buffer[split_idx+1:]
                                             except ValueError: break # No sentence end found
                                     # Check if stream is done *after* processing potential chunk
                                     if data.get('done'):
                                         if buffer: yield buffer # Yield remaining buffer
                                         break
                                 except json.JSONDecodeError: logger.warning(f"Failed to decode JSON line: {line}"); yield "[Error: Invalid response chunk]"
                                 except Exception as stream_e: logger.error(f"Error processing stream chunk: {stream_e}"); yield f"[Error: {stream_e}]"
                         return # End stream generator

                    else: # Non-streaming
                         try:
                             data = await response.json()
                             response_text = data.get('response', '').strip()
                             if response_text: yield response_text
                             else: logger.warning("Ollama non-stream response was empty."); yield "[Warning: LLM returned empty response]"
                         except json.JSONDecodeError: logger.error(f"Failed to decode non-stream JSON response."); yield "[Error: Invalid LLM response format]"
                         return # End non-stream generator (yielded one item)

        except aiohttp.ClientConnectorError as e: logger.error(f"Cannot connect to Ollama at {self.ollama_base_url}: {e}"); yield "[Error: Cannot connect to Ollama]"
        except asyncio.TimeoutError: logger.error(f"Timeout connecting to Ollama"); yield "[Error: Ollama request timed out]"
        except Exception as e: logger.error(f"Ollama generate call failed: {e}", exc_info=True); yield f"[Error: {e}]"
        # Generator finishes naturally
