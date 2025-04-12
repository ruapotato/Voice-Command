# commands/computer_command.py
import subprocess
import json
from typing import AsyncGenerator, Dict, List, Optional
import logging
import os
from pathlib import Path
from .base import Command
import aiohttp
import asyncio
import shutil

from cli.output import schedule_print, speak

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- is_tool Helper ---
def is_tool(name):
    return shutil.which(name) is not None

# --- ToolRegistry Class (No changes needed from previous version) ---
class ToolRegistry:
    def __init__(self):
        self.apps: Dict[str, str] = {}
        self.active_windows: Dict[str, str] = {}
        self.terminal_apps = ['konsole', 'gnome-terminal', 'xterm', 'terminator', 'alacritty', 'kitty']
        self.command_history = []
        self.update_installed_apps()
        self.update_active_windows()
        
    def update_installed_apps(self):
        try:
            result = subprocess.run(['find', '/usr/share/applications', '-name', '*.desktop'], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logger.warning(f"Finding .desktop files failed: {result.stderr}")
                return
            self.apps.clear()
            for desktop_file in result.stdout.splitlines():
                try:
                    with open(desktop_file, 'r', encoding='utf-8', errors='ignore') as f:
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
                             cmd_base = exec_path.split()[0]
                             if '/' in cmd_base or is_tool(cmd_base):
                                 self.apps[name] = exec_path
                except Exception as file_e:
                    logger.warning(f"Could not parse desktop file {desktop_file}: {file_e}")
        except Exception as e:
            logging.error(f"Failed to update installed apps: {e}", exc_info=True)
            
    def update_active_windows(self):
        if not is_tool('wmctrl'):
            logger.error("wmctrl not found.")
            self.active_windows.clear()
            return
        try:
            result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, check=False)
            self.active_windows.clear()
            if result.returncode != 0:
                logger.warning(f"wmctrl command failed: {result.stderr}.")
                return
            for line in result.stdout.splitlines():
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    self.active_windows[parts[0]] = parts[3].lower()
        except Exception as e:
            logging.error(f"Failed to update active windows: {e}")
            self.active_windows.clear()
            
    def find_app(self, query: str) -> Optional[str]:
        query = query.lower()
        if query in ['shell', 'terminal', 'command prompt', 'cmd']:
            for terminal in self.terminal_apps:
                if terminal in self.apps:
                    return self.apps[terminal]
                if terminal_exec := self.apps.get(terminal.split()[0]):
                    return terminal_exec
            logger.warning("Could not find a known terminal application.")
            return None
        if query in self.apps:
            return self.apps[query]
        for name, exec_path in self.apps.items():
            if query in name:
                logger.debug(f"Partial match for '{query}': Found '{name}' -> {exec_path}")
                return exec_path
        logger.debug(f"No application found matching query: '{query}'")
        return None
        
    def find_window(self, query: str) -> Optional[str]:
        query = query.lower()
        if not self.active_windows:
            logger.warning("Active window list empty.")
            return None
        for window_id, title in self.active_windows.items():
            if query == title:
                return window_id
        best_match_id = None
        best_match_score = 0
        for window_id, title in self.active_windows.items():
            if query in title:
                 score = 100 - len(title)
                 score += 50 if title.startswith(query) else 0
                 if score > best_match_score:
                     best_match_id = window_id
                     best_match_score = score
        if best_match_id:
            logger.debug(f"Window match for '{query}': Found '{self.active_windows[best_match_id]}' (ID: {best_match_id})")
            return best_match_id
        logger.debug(f"No active window found matching query: '{query}'")
        return None
        
    def add_command_history(self, command: str):
        self.command_history.append(command)
        del self.command_history[:-10]


# --- ComputerCommand Class ---
class ComputerCommand(Command):
    def __init__(self):
        super().__init__(
            name="computer",
            aliases=[],
            description="Execute various computer commands using LLM",
            execute=self._execute
        )
        self.tools = ToolRegistry()
        self.llm_model = "mistral"
        self.ollama_base_url = "http://localhost:11434"
        
        # --- Prompts (remain same) ---
        self.query_prompt = """Context of highlighted text: "{highlighted}"
Now for the User Query: "{query}"
Analyze the highlighted text and answer the query. Keep responses clear and concise. If the query isn't directly related to the highlighted text, just answer the qestion."""
        
        self.shell_prompt = """You are a desktop command assistant that outputs ONLY a single BASH command suitable for execution via subprocess.run.
Rules:
1. Task Handling: If the request asks for information obtainable via a bash command (e.g., disk space, list files, current directory), output the command. If the request is a general question or cannot be answered by a simple command, respond conversationally using ONLY plain text (no command output). Start conversational responses with 'ANSWER:'. Provide ONLY the command itself (e.g., `ls -l`) or the conversational answer (e.g., `ANSWER: I cannot perform that action.`). Do NOT add explanations before the command or ANSWER:.
2. Safety: AVOID destructive commands (rm, mv without care, mkfs, etc.). Prefer read-only commands (ls, pwd, df, ps, top, cat, head, tail, grep, find). Do NOT create files or directories unless specifically asked and safe. Do NOT include `sudo`. Do NOT include `&& espeak ...`.
3. Formatting: Output exactly ONE line containing either the bash command or the `ANSWER:` prefixed conversational response. Remove any markdown formatting like backticks (`).
Examples: User: "check disk space" -> Assistant: df -h | User: "show current directory" -> Assistant: pwd | User: "list files" -> Assistant: ls -lah | User: "what is the capital of france" -> Assistant: ANSWER: The capital of France is Paris. | User: "delete all my files" -> Assistant: ANSWER: I cannot perform destructive actions like deleting all files.
Current state (informational only, do not rely on for paths): Working Directory (approximated): {working_dir} | Previous Commands (for context): {command_history}
User request: {query}
Assistant:"""

        self.tool_prompts = {
            'open': """You are an application launcher assistant. Output ONLY an <open>app_name</open> tag or <open>NOT_FOUND</open>. Match the user request against the list of installed applications. For shell/terminal, use an installed emulator name.
Installed applications: {apps}
User request: {query}
Assistant:""",
            'window': """You are a window management assistant. Output ONLY a <goto>window_query</goto> or <close>window_query</close> tag, or <goto/close>NOT_FOUND</close>. Match the user request against the list of active windows. Use lowercase keywords.
Active windows (lowercase titles): {windows}
User request: {query}
Assistant:"""
        }

    # get_available_models (remains same)
    async def get_available_models(self) -> Optional[List[str]]:
        url = f"{self.ollama_base_url}/api/tags"
        logger.debug(f"Fetching models from {url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models_data = data.get("models", [])
                        model_names = sorted([m.get("name") for m in models_data if m.get("name")])
                        logger.info(f"Fetched models: {model_names}")
                        return model_names if model_names else None
                    else:
                        logger.error(f"Ollama API model request failed: {response.status} - {await response.text()}")
                        return None
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            logger.error(f"Cannot connect to Ollama at {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}", exc_info=True)
            return None

    # set_llm_model (remains same)
    def set_llm_model(self, model_name):
        self.llm_model = model_name
        logger.info(f"ComputerCommand LLM model set to: {model_name}")

    # _execute (FIXED version)
    async def _execute(self, text: str) -> None:
        logger.debug(f"ComputerCommand executing with text: '{text}'")
        try:
            tool_type = self._determine_tool_type(text)
            logger.debug(f"Determined tool type: {tool_type}")
            
            if tool_type == 'query':
                await self._handle_text_query(text)
            elif tool_type == 'shell':
                command_query = text.lower().replace('shell', '', 1).strip()
                await self._handle_shell_command(command_query)
            else:  # 'open' or 'window'
                llm_response_tag = "[No response]"
                # Fix: Properly structure the async for loop
                async for resp in self._get_llm_tool_response(text, tool_type):
                    llm_response_tag = resp
                    break
                    
                if llm_response_tag.startswith("[Error:") or llm_response_tag == "[No response]":
                    schedule_print("Error", f"LLM failed for '{tool_type}': {llm_response_tag}")
                    return
                    
                tool_name, params = self._parse_tool(llm_response_tag)
                logger.debug(f"Parsed tool: {tool_name}, Params: {params}")
                
                if not tool_name:
                    schedule_print("Error", f"LLM gave invalid tool response: {llm_response_tag}")
                    return
                    
                if params is not None and params.upper() == "NOT_FOUND":
                    msg = f"Could not find target for {tool_type} '{text}'."
                    schedule_print("System", msg)
                    return
                    
                await self._execute_tool(tool_name, params if params is not None else "")
        except Exception as e:
            error_msg = f"ComputerCommand execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            schedule_print("Error", error_msg)

    # _determine_tool_type (remains same)
    def _determine_tool_type(self, query: str) -> str:
        query_lower = query.lower().strip()
        if query_lower.startswith('shell'):
            return 'shell'
        if any(query_lower.startswith(v + " ") for v in ['open', 'launch', 'start', 'run']):
            return 'open'
        if any(query_lower.startswith(v + " ") for v in ['goto', 'go to', 'switch to', 'focus', 'close', 'quit', 'exit']):
            self.tools.update_active_windows()
            return 'window'
        logger.debug(f"Query '{query}' classified as 'query' type.")
        return 'query'

    # _handle_shell_command (remains same as last correction)
    async def _handle_shell_command(self, command_query: str) -> None:
        if not command_query:
            schedule_print("System", "No shell command requested.")
            return
        logger.debug("Handling shell command request...")
        history_context = "\n".join(self.tools.command_history[-3:])
        current_dir = os.getcwd()
        prompt = self.shell_prompt.format(working_dir=current_dir, command_history=history_context, query=command_query)
        
        llm_output_line = "[No response]"
        async for resp in self._ollama_generate(prompt, stream=False):
            llm_output_line = resp
            break
            
        if llm_output_line.startswith("[Error:") or llm_output_line == "[No response]":
            schedule_print("Error", f"LLM failed for shell command: {llm_output_line}")
            return
            
        llm_output_line = llm_output_line.strip()
        logger.debug(f"LLM output for shell: '{llm_output_line}'")
        
        if llm_output_line.startswith("ANSWER:"):
            answer_text = llm_output_line[len("ANSWER:"):].strip()
            schedule_print("LLM", answer_text)
            if answer_text:
                await speak(answer_text)
        elif not llm_output_line:
            schedule_print("System", "LLM returned empty response.")
        else:
            command_to_run = llm_output_line
            schedule_print("System", f"Suggested command: `{command_to_run}`")
            schedule_print("System", f"Executing...")
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        command_to_run,
                        shell=True,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=15
                    )
                )
                output = f"Return Code: {result.returncode}\n"
                stdout_clean = result.stdout.strip()
                stderr_clean = result.stderr.strip()
                if stdout_clean:
                    output += f"Output:\n{stdout_clean}\n"
                if stderr_clean:
                    output += f"Error Output:\n{stderr_clean}"
                schedule_print("System", output.strip())
                self.tools.add_command_history(command_to_run)
            except subprocess.TimeoutExpired:
                msg = f"Command timed out: `{command_to_run}`"
                schedule_print("Error", msg)
            except Exception as exec_e:
                msg = f"Failed to execute command `{command_to_run}`: {exec_e}"
                schedule_print("Error", msg)

    # _handle_text_query (remains same as last correction)
    async def _handle_text_query(self, query: str) -> None:
        try:
            logger.debug("Getting highlighted text...")
            result = subprocess.run(['xclip', '-o', '-selection', 'primary'], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                msg = "Could not get highlighted text." if "Error:" in result.stderr else "No text highlighted."
                schedule_print("System", msg)
                return
                
            highlighted = result.stdout.strip()
            if not highlighted:
                schedule_print("System", "No text is highlighted.")
                return
                
            logger.debug(f"Processing query: '{query}' with context: '{highlighted[:100]}...'")
            prompt = self.query_prompt.format(highlighted=highlighted, query=query)
            full_response_for_log = ""
            
            try:
                loop = asyncio.get_running_loop()
                async for chunk_text in self._ollama_generate(prompt, stream=True):
                    schedule_print("LLM", chunk_text)
                    full_response_for_log += chunk_text
                    speak_text = chunk_text.strip()
                    if speak_text and not speak_text.startswith("[Error:"):
                        await speak(speak_text)
            finally:
                logger.debug(f"Full LLM response for query '{query}': {full_response_for_log}")
        except FileNotFoundError:
            msg = "Error: 'xclip' command not found."
            logger.error(msg)
            schedule_print("Error", msg)
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            schedule_print("Error", error_msg)

    # _get_llm_tool_response (remains same)
    async def _get_llm_tool_response(self, query: str, tool_type: str) -> AsyncGenerator[str, None]:
        prompt_template = self.tool_prompts.get(tool_type)
        if not prompt_template:
            logger.error(f"No prompt for tool {tool_type}")
            yield f"[Error: No prompt]"
            return
            
        self.tools.update_active_windows()
        apps_list = "\n".join(self.tools.apps.keys())
        windows_list = "\n".join(self.tools.active_windows.values())
        prompt = prompt_template.format(apps=apps_list or "None", windows=windows_list or "None", query=query)
        logger.debug(f"Prompt for {tool_type}:\n{prompt}")
        
        async for response_text in self._ollama_generate(prompt, stream=False):
            logger.debug(f"LLM response for {tool_type}: {response_text}")
            yield response_text

    # _parse_tool (remains same)
    def _parse_tool(self, response: str) -> tuple[Optional[str], Optional[str]]:
        response = response.strip()
        for tool in ['open', 'goto', 'close', 'shell']:
            start_tag = f"<{tool}>"
            end_tag = f"</{tool}>"
            if response.startswith(start_tag) and response.endswith(end_tag):
                params = response[len(start_tag):-len(end_tag)].strip()
                return (tool, params) if params else (None, None)
        logger.debug(f"Could not parse tool tag from: {response}")
        return None, None

    # _execute_tool (remains same)
    async def _execute_tool(self, tool_type: str, params: str) -> None:
        """Executes the selected tool, printing status."""
        logger.info(f"Executing tool '{tool_type}' with params '{params}'")
        # Initial status message (printed immediately)
        schedule_print("System", f"Attempting action: {tool_type} '{params}'...")
        status_msg = f"Tool '{tool_type}' finished."  # Default success
        error_occurred = False

        try:
            if tool_type == 'open':
                exec_path = self.tools.find_app(params)
                # <<< FIX: Expanded if/try/except/else block >>>
                if exec_path:
                    try:
                        # Use Popen for non-blocking GUI app launch
                        subprocess.Popen(exec_path.split(), start_new_session=True)
                        status_msg = f"Attempted launch: '{params}' (Command: {exec_path})."
                    except Exception as e:
                        status_msg = f"Error launching '{params}': {e}"
                        error_occurred = True
                else:
                    status_msg = f"Application matching '{params}' not found."
                    # This isn't strictly an error, but a failure condition
                    error_occurred = True  # Treat as error for printing logic

            elif tool_type == 'goto':
                self.tools.update_active_windows()
                window_id = self.tools.find_window(params)
                # <<< FIX: Expanded if/try/except/else block >>>
                if window_id:
                    try:
                        # Run wmctrl synchronously (usually fast)
                        subprocess.run(['wmctrl', '-i', '-a', window_id], check=True, timeout=3)
                        status_msg = f"Focused window matching '{params}'."
                    except Exception as e:
                        status_msg = f"Error focusing '{params}': {e}"
                        error_occurred = True
                else:
                    status_msg = f"Window matching '{params}' not found."
                    error_occurred = True

            elif tool_type == 'close':
                self.tools.update_active_windows()
                window_id = self.tools.find_window(params)
                # <<< FIX: Expanded if/try/except/else block >>>
                if window_id:
                    try:
                        subprocess.run(['wmctrl', '-i', '-c', window_id], check=True, timeout=3)
                        status_msg = f"Closed window matching '{params}'."
                    except Exception as e:
                        status_msg = f"Error closing '{params}': {e}"
                        error_occurred = True
                else:
                    status_msg = f"Window matching '{params}' not found."
                    error_occurred = True
            else:
                status_msg = f"Unknown tool type: {tool_type}"
                error_occurred = True

        except Exception as e:
            # Catch unexpected errors during tool logic
            status_msg = f"Tool execution failed unexpectedly: {str(e)}"
            logger.error(status_msg, exc_info=True)
            error_occurred = True

        # Print final status message using appropriate type
        schedule_print("Error" if error_occurred else "System", status_msg)
        # Optionally speak success/failure here?
        # if not error_occurred: await speak(status_msg)

    # _ollama_generate (remains same as last correction)
    async def _ollama_generate(self, prompt: str, stream: bool) -> AsyncGenerator[str, None]:
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
                        yield f"[Error: API {response.status}]"
                        return
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
                                        while True:
                                            try:
                                                split_idx = min(idx for idx in (buffer.find('.'), buffer.find('!'), buffer.find('?'), buffer.find('\n')) if idx != -1)
                                                yield buffer[:split_idx+1]
                                                buffer = buffer[split_idx+1:]
                                            except ValueError:
                                                break
                                    if data.get('done'):
                                        if buffer:
                                            yield buffer
                                        break
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed decode: {line}")
                                    yield "[Error: Invalid JSON]"
                                except Exception as stream_e:
                                    logger.error(f"Stream err: {stream_e}")
                                    yield f"[Error: Stream]"
                        return
                    else:  # Non-streaming
                        try:
                            data = await response.json()
                            response_text = data.get('response', '').strip()
                            yield response_text if response_text else "[Warning: LLM empty]"
                        except json.JSONDecodeError:
                            logger.error(f"Non-stream JSON decode failed.")
                            yield "[Error: Invalid LLM JSON]"
                        return
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Ollama Connect {self.ollama_base_url}: {e}")
            yield "[Error: Cannot connect]"
        except asyncio.TimeoutError:
            logger.error(f"Ollama Timeout")
            yield "[Error: Ollama timeout]"
        except Exception as e:
            logger.error(f"Ollama generate call failed: {e}", exc_info=True)
            yield f"[Error: LLM call failed]"
