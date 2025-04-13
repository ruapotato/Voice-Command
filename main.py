#!/usr/bin/env python3

import asyncio
import sys
import logging
from typing import List, Optional, Callable, Any, Coroutine
import textwrap
import re # <<< For input normalization

# Third-party imports
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
# Optional: For history and suggestions
# from prompt_toolkit.history import FileHistory
# from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

# Project imports
from cli.output import schedule_print, print_consumer, print_queue, safe_print, speak
from cli.completer import CLICompleter, ollama_models_for_completion
import hotkey_listener # Keep import
from commands.command_processor import CommandProcessor
from commands.computer_command import ComputerCommand # Keep direct import if needed
from core.voice_system import VoiceCommandSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
ollama_models_list: List[str] = ["mistral"] # Default model
computer_command_instance: Optional[ComputerCommand] = None
# --- Global for tracking the current command task ---
current_command_task: Optional[asyncio.Task] = None

# --- Accessor function for the hotkey listener ---
def get_current_task() -> Optional[asyncio.Task]:
    """Returns the currently active command task, if any."""
    global current_command_task
    return current_command_task

# --- Voice Command Handler ---
async def handle_voice_command(text: str, command_processor: CommandProcessor):
    """Processes voice commands, manages the task, handles cancellation."""
    global current_command_task
    if current_command_task and not current_command_task.done():
        logger.warning("New voice command received while previous task running. Cancelling previous.")
        current_command_task.cancel()
        try:
            # Give cancellation a chance to propagate
            await asyncio.wait_for(current_command_task, timeout=0.5)
        except asyncio.CancelledError:
            logger.debug("Previous voice task cancelled successfully.")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for previous voice task cancellation.")
        except Exception as e:
            logger.error(f"Error awaiting previous cancelled voice task: {e}")
        finally:
             current_command_task = None # Explicitly clear

    logger.info(f"Processing voice command: {text}")
    # Create a task for the command processing using the helper
    current_command_task = asyncio.create_task(
        _execute_command_stream(text, command_processor, "Voice")
    )

    try:
        await current_command_task
    except asyncio.CancelledError:
        # Message is printed by the interrupt handler or the task itself
        logger.info("Voice command task was cancelled.")
    except Exception as e:
        logger.error(f"Error executing voice command task '{text}': {e}", exc_info=True)
        schedule_print("Error", f"Failed processing voice command: {e}")
    finally:
        # Ensure the task reference is cleared after completion/cancellation
        if current_command_task and current_command_task.done():
             current_command_task = None

# --- Transcript Callback (for printing only) ---
def handle_transcript(text: str, source: str = "Voice"):
    """Callback from Voice System. Schedules print updates."""
    schedule_print(source, text)

# --- Typed Command Processing ---
async def process_typed_command(text: str, command_processor: CommandProcessor):
    """Processes commands entered via the CLI, manages the task, handles cancellation."""
    global current_command_task
    if current_command_task and not current_command_task.done():
        logger.warning("New typed command received while previous task running. Cancelling previous.")
        current_command_task.cancel()
        try:
             # Give cancellation a chance to propagate
            await asyncio.wait_for(current_command_task, timeout=0.5)
        except asyncio.CancelledError:
            logger.debug("Previous typed task cancelled successfully.")
        except asyncio.TimeoutError:
             logger.warning("Timeout waiting for previous typed task cancellation.")
        except Exception as e:
            logger.error(f"Error awaiting previous cancelled typed task: {e}")
        finally:
            current_command_task = None # Explicitly clear


    schedule_print("Typed", text)
    # Create a task for the command processing using the helper
    current_command_task = asyncio.create_task(
        _execute_command_stream(text, command_processor, "Typed")
    )

    try:
        await current_command_task
    except asyncio.CancelledError:
        # Message is printed by the interrupt handler or the task itself
        logger.info("Typed command task was cancelled.")
    except Exception as e:
        logger.error(f"Error executing typed command task '{text}': {e}", exc_info=True)
        schedule_print("Error", f"Failed processing command: {e}")
    finally:
        # Ensure the task reference is cleared after completion/cancellation
        if current_command_task and current_command_task.done():
             current_command_task = None

# --- Helper for executing commands and handling streams/speaking ---
async def _execute_command_stream(text: str, command_processor: CommandProcessor, source: str):
    """Internal helper to run command processor and handle output/speech."""
    original_cmd_name = None # Keep track of the originally matched command
    try:
        # <<< Normalize the input text FIRST >>>
        # Lowercase, strip leading/trailing whitespace
        normalized_text = text.lower().strip()
        # Remove common trailing punctuation (periods, question marks, exclamation points)
        # using regex for cleaner removal than multiple .rstrip() calls.
        normalized_text = re.sub(r'[.!?]+$', '', normalized_text).strip()
        # Example: "Read." -> "read", "Computer what is this?" -> "computer what is this"
        # Keep the original 'text' variable intact in case we need it for typing.
        logger.debug(f"Original text: '{text}', Normalized for parsing: '{normalized_text}'")
        # <<< END Normalization >>>

        # Parse the NORMALIZED text to find the command
        cmd_name, args = command_processor.parse_command(normalized_text)
        original_cmd_name = cmd_name # Store the matched command name

        processor_input = normalized_text # Start with normalized for command execution by default
        is_general_query = False

        if cmd_name:
            # --- Specific command matched ---
            schedule_print("System", f"Executing: {cmd_name} {args if args else ''}")
            # processor_input is already normalized_text, which includes args if found by parser
            is_general_query = (cmd_name == "computer")
        else:
            # --- No specific command matched: Treat as Type command ---
            logger.info(f"No command keyword matched for normalized input: '{normalized_text}'. Treating as 'type' command.")
            schedule_print("System", f"No command matched. Typing...") # Update printed message

            # IMPORTANT: Use the ORIGINAL, un-normalized 'text' for typing
            # so that punctuation and capitalization are preserved.
            processor_input = f"type {text}"
            original_cmd_name = "type" # Update for speaking logic
            is_general_query = False

        # --- Execute the command (either original or the reconstructed 'type' command) ---
        # Pass the appropriate input (normalized for commands, reconstructed 'type' for fallback)
        async for result in command_processor.process_command(processor_input):
            # Determine message type
            msg_type = "LLM" if is_general_query else "System"
            schedule_print(msg_type, f"{result}") # Always print the result

            # --- Speaking Logic ---
            should_speak = False # Default to not speaking
            if result and not isinstance(result, (list, dict)):
                result_str = str(result)
                # Speak only if it's NOT from 'read', 'computer', 'type', 'stop', etc.
                # And not an error or common status message.
                if original_cmd_name not in ["read", "computer", "type", "stop"] and \
                   not result_str.startswith("[Error:") and \
                   not result_str.startswith("Suggested command:") and \
                   not result_str.startswith("Attempting action:") and \
                   not result_str.startswith("Interrupted") and \
                   not result_str.startswith("Finished reading") and \
                   not result_str.startswith("Typed:") and \
                   not result_str.startswith("Read command executed.") and \
                   not result_str.startswith("No command matched."):
                     should_speak = True

            if should_speak:
                await speak(result_str) # Call global speak function
            # --- End Speaking Logic ---

    except asyncio.CancelledError:
         # Log cancellation specifically for this execution context
         logger.info(f"Command execution for '{text}' cancelled within stream helper.")
         # Don't schedule print here, handled by caller or interrupt handler
         raise # Re-raise cancellation error to be caught by the caller
    except Exception as e:
        # Log and schedule print for errors during command execution itself
        logger.error(f"Error processing command '{text}' in _execute_command_stream: {e}", exc_info=True)
        schedule_print("Error", f"Failed processing command: {e}")
        # Don't re-raise normal exceptions, let the task finish with error logged.


# --- Help Text Generation ---
def generate_help_text(command_processor: CommandProcessor) -> str:
    """Generates help text dynamically from registered commands."""
    lines = []; lines.append("Available commands:")
    details = command_processor.get_command_details(); max_len = 0
    if details:
        # Calculate max length for alignment
        for name, aliases, _ in details:
             alias_str = f" ({', '.join(aliases)})" if aliases else ""
             max_len = max(max_len, len(name) + len(alias_str))
    static_cmds = ["select model [name]", "refresh_models", "help", "exit / quit"] # Removed 'stop'
    if static_cmds: # Avoid error if list is empty
         max_len = max(max_len, max(len(s) for s in static_cmds))

    indent = "  "; padding = 2; desc_width = 70 # Adjust desc_width if needed

    # Add registered commands details
    for name, aliases, description in sorted(details, key=lambda x: x[0]): # Sort commands alphabetically
        alias_str = f" ({', '.join(aliases)})" if aliases else ""
        command_part = f"{indent}{name}{alias_str}".ljust(max_len + len(indent) + padding)
        # Wrap description
        wrapped_desc = textwrap.wrap(description or "No description.", width=desc_width)
        # Add first line of description (or only line)
        lines.append(f"  <ansiblue>{command_part}</ansiblue> {wrapped_desc[0] if wrapped_desc else ''}")
        # Add subsequent lines of description indented
        for line in wrapped_desc[1:]:
            lines.append(f"{indent}{' ' * (max_len + len(indent) + padding)} {line}")

    lines.append("\nOther CLI Commands:")
    lines.append(f"{indent}{'select model [name]'.ljust(max_len + padding)} - Switch the Ollama LLM model.")
    lines.append(f"{indent}{'refresh_models'.ljust(max_len + padding)} - Reload list of available Ollama models.")
    # lines.append(f"{indent}{'stop'.ljust(max_len + padding)} - Stops active text-to-speech feedback (use Ctrl+C).") # Optional: Keep or remove stop command
    lines.append(f"{indent}{'help'.ljust(max_len + padding)} - Shows this help message.")
    lines.append(f"{indent}{'exit / quit'.ljust(max_len + padding)} - Exits the application.")

    lines.append("\nUsage:")
    # --- Updated Usage Section ---
    lines.append(f"{indent}- Start input with a known command keyword (e.g., 'click OK', 'read', 'screengrab')")
    lines.append(f"{indent}  to execute that specific command.")
    lines.append(f"{indent}- To query the LLM, you MUST start with the 'computer' keyword")
    lines.append(f"{indent}  (e.g., 'computer what is the weather in Sutherlin, Oregon?').") # Added location
    lines.append(f"{indent}- Any input (voice or typed) that DOES NOT start with a known command keyword")
    lines.append(f"{indent}  will be automatically TYPED out, similar to the 'type' command.")
    lines.append(f"{indent}  Example: Saying 'hello world' will result in 'hello world' being typed.")
    lines.append(f"\n{indent}Hotkeys:")
    lines.append(f"{indent}- Voice Activation: Press and hold Ctrl+Alt to record voice input.")
    lines.append(f"{indent}- Interruption: Press Ctrl+C to stop the current command or speech output.")
    # --- End Updated Usage Section ---
    return "\n".join(lines)

# --- Dynamic Prompt Function ---
def get_dynamic_prompt() -> str:
    """Returns the prompt string including the current model."""
    model = computer_command_instance.llm_model if computer_command_instance else "???"
    # Changed prompt slightly to indicate default action
    return f"Cmd/Type ({model})> "

# --- Main Application Logic ---
async def async_main():
    """Main asynchronous function for the CLI."""
    global ollama_models_list, computer_command_instance, current_command_task

    main_event_loop = asyncio.get_running_loop()
    print_task = asyncio.create_task(print_consumer())
    # Allow print consumer to start up
    await asyncio.sleep(0.05)

    schedule_print("System", "Initializing Voice Command CLI...")

    # Initialize Command Processor
    try:
        command_processor = CommandProcessor()
        computer_command_instance = command_processor.commands.get("computer")
    except Exception as e:
        logger.critical(f"Failed to initialize CommandProcessor: {e}", exc_info=True)
        schedule_print("Error", f"CRITICAL: Failed to load commands: {e}")
        # Attempt cleanup and exit
        await print_queue.put((None, None))
        try: await asyncio.wait_for(print_task, timeout=1.0)
        except Exception: pass
        sys.exit(1)


    # --- Initialize Ollama Models ---
    if not isinstance(computer_command_instance, ComputerCommand):
        schedule_print("Warning", "ComputerCommand module not loaded correctly. LLM features disabled.")
    else:
        schedule_print("System", "Fetching Ollama models...")
        try:
            fetched_models = await computer_command_instance.get_available_models()
            if fetched_models: # Successfully fetched list (might be empty)
                ollama_models_list = fetched_models
                ollama_models_for_completion[:] = ollama_models_list
                if ollama_models_list: # List is not empty
                    # Check if current model (default 'mistral') is valid, else use first available
                    current_model = computer_command_instance.llm_model # Get current default
                    if current_model not in ollama_models_list:
                        new_model = ollama_models_list[0]
                        computer_command_instance.set_llm_model(new_model)
                        schedule_print("System", f"Default model '{current_model}' not found. Switched to: {new_model}")
                    else:
                        schedule_print("System", f"Ollama models loaded. Using: {current_model}")

                else: # List is empty
                    schedule_print("Warning", "No Ollama models found. LLM commands may fail.")
                    computer_command_instance.set_llm_model("mistral") # Keep fallback
            else: # API call failed (returned None)
                schedule_print("Warning", f"Could not fetch Ollama models (API error?). Using default: {ollama_models_list[0]}")
                ollama_models_for_completion[:] = ollama_models_list # Update completer with default
                computer_command_instance.set_llm_model(ollama_models_list[0]) # Ensure default is set
        except Exception as e:
            schedule_print("Error", f"Failed during Ollama model fetch: {e}. Using default.")
            ollama_models_for_completion[:] = ollama_models_list # Update completer with default
            if computer_command_instance: computer_command_instance.set_llm_model(ollama_models_list[0])


    # --- Initialize Voice System & Hotkey ---
    voice_system = None # Define voice_system before try block
    listener_thread = None # Define listener_thread before try block

    # --- Define the callback function that VoiceCommandSystem will call ---
    async def trigger_command_processing(transcribed_text: str):
        """Callback from Voice System to handle final transcription."""
        # This function runs in the main async context
        await handle_voice_command(transcribed_text, command_processor)

    try:
        # Pass the new trigger function during initialization
        voice_system = VoiceCommandSystem(
             loop=main_event_loop,
             speak_func=speak,
             command_trigger_func=trigger_command_processing # Pass the async callback
        )
        # Set the callback for printing transcripts/status
        voice_system.set_transcript_callback(handle_transcript)
        schedule_print("System", "Voice system initialized.")

        # Pass the task accessor function to the listener
        listener_thread = hotkey_listener.start_listener(
             main_event_loop,
             voice_system,
             schedule_print,
             get_current_task # Pass the accessor function
        )
        if not listener_thread:
             raise RuntimeError("Hotkey listener failed to start.")

    except Exception as e:
        logger.error(f"Failed to initialize VoiceCommandSystem or Hotkey Listener: {e}", exc_info=True)
        schedule_print("Error", "Failed to initialize voice system or hotkey. Voice commands/hotkeys disabled.")
        if voice_system: # Attempt cleanup if voice system partially initialized
             try:
                 # Check if cleanup is async or sync
                 if asyncio.iscoroutinefunction(voice_system.cleanup):
                     await voice_system.cleanup()
                 else:
                     # Run synchronous cleanup in executor if needed, or directly if safe
                     await main_event_loop.run_in_executor(None, voice_system.cleanup)
             except Exception as cleanup_e: logger.error(f"Error during voice system cleanup after init failure: {cleanup_e}")
        voice_system = None # Ensure voice_system is None if init fails


    # --- Setup Prompt Session ---
    cli_completer = CLICompleter(command_processor)
    session = PromptSession(
        get_dynamic_prompt, # Dynamic prompt function
        completer=cli_completer,
        complete_while_typing=True,
        # history=FileHistory('cli_history.txt'), # Optional: Uncomment for history
        # auto_suggest=AutoSuggestFromHistory(), # Optional: Uncomment for suggestions
    )

    schedule_print("System", f"CLI Ready. Type 'help' for commands or use hotkeys.")

    # --- Main Input Loop ---
    while True:
        input_text = "" # Ensure defined in outer scope
        try:
            # Use patch_stdout to ensure prompt redraws correctly after async prints
            with patch_stdout():
                input_text = await session.prompt_async() # Use await

            input_text = input_text.strip()
            if not input_text: continue # Ignore empty input

            # --- Handle Special CLI Commands ---
            if input_text.lower() in ["exit", "quit"]:
                schedule_print("System", "Exiting...")
                break # Exit the main loop

            elif input_text.lower() == "help":
                help_content = generate_help_text(command_processor)
                # Use safe_print directly for potentially long help text with formatting
                await safe_print(help_content)
                continue

            elif input_text.lower() == "refresh_models":
                if computer_command_instance:
                    schedule_print("System", "Refreshing Ollama models...")
                    try:
                        fetched_models = await computer_command_instance.get_available_models()
                        if fetched_models is not None: # Check for None explicitly
                            ollama_models_list = fetched_models
                            ollama_models_for_completion[:] = ollama_models_list
                            if not ollama_models_list: # List could be empty
                                 schedule_print("Warning", "No Ollama models found after refresh.")
                                 if computer_command_instance.llm_model not in ollama_models_list:
                                     computer_command_instance.set_llm_model("mistral") # Fallback
                                     schedule_print("System", "Model reset to fallback 'mistral'.")
                            elif computer_command_instance.llm_model not in ollama_models_list:
                                new_model = ollama_models_list[0] # Use first available
                                computer_command_instance.set_llm_model(new_model)
                                schedule_print("System", f"Models refreshed: {ollama_models_list}. Current model reset to {new_model}")
                            else:
                                schedule_print("System", f"Models refreshed: {ollama_models_list}")
                        else: # get_available_models returned None (error occurred)
                             schedule_print("Error", "Failed to fetch models (API error or connection issue).")
                    except Exception as e: schedule_print("Error", f"Failed to refresh models: {e}")
                else: schedule_print("Error", "Computer command module unavailable.")
                continue

            elif input_text.lower().startswith("select model "):
                parts = input_text.split(maxsplit=2)
                if len(parts) == 3:
                    model_name = parts[2]
                    # Check against the potentially empty list
                    if ollama_models_list and model_name in ollama_models_list:
                        if computer_command_instance:
                             computer_command_instance.set_llm_model(model_name)
                             schedule_print("System", f"LLM model set to: {model_name}")
                        else: schedule_print("Error", "Computer command module unavailable.")
                    elif not ollama_models_list:
                         schedule_print("Error", "No models available to select.")
                    else:
                         # Provide available models in error message
                         available_models_str = ', '.join(ollama_models_list)
                         schedule_print("Error", f"Model '{model_name}' not found. Available: {available_models_str}")
                else: schedule_print("Error", "Usage: select model <model_name>")
                continue

            # --- Process Regular Commands / Default Typing ---
            await process_typed_command(input_text, command_processor)

        except KeyboardInterrupt:
            # Handle Ctrl+C pressed *at the prompt*
            if current_command_task and not current_command_task.done():
                 logger.debug("Ctrl+C at prompt: Cancelling active task.")
                 current_command_task.cancel()
                 # Interrupt handler will print the message
            else:
                 # Schedule print message for Ctrl+C when idle if needed
                 # schedule_print("System", "(Ctrl+C at prompt)") # Can be noisy
                 pass # Just redraw prompt by continuing
            # Continue to redraw prompt
            continue
        except EOFError:
            # Exit gracefully on Ctrl+D
            schedule_print("System", "EOF received. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error in main loop processing input '{input_text}': {e}", exc_info=True)
            schedule_print("Error", f"An unexpected error occurred: {e}")
            # Prevent rapid error loops if prompt_async fails repeatedly
            await asyncio.sleep(0.1)

    # --- Application Cleanup ---
    schedule_print("System", "Shutting down...")

    # --- Cancel any lingering task ---
    if current_command_task and not current_command_task.done():
         logger.info("Shutting down: Cancelling active command task.")
         current_command_task.cancel()
         try:
             await asyncio.wait_for(current_command_task, timeout=1.0) # Wait briefly
         except asyncio.CancelledError: pass # Expected
         except asyncio.TimeoutError: logger.warning("Timeout waiting for final task cancellation.")
         except Exception as e: logger.error(f"Error awaiting final task cancellation: {e}")


    # --- Voice System Cleanup ---
    if voice_system and hasattr(voice_system, 'cleanup'):
        logger.info("Calling voice system cleanup...")
        try:
            # Check if cleanup is async or sync
            if asyncio.iscoroutinefunction(voice_system.cleanup):
                await voice_system.cleanup()
            else:
                # Run synchronous cleanup in executor if needed, or directly if safe
                await main_event_loop.run_in_executor(None, voice_system.cleanup)
            logger.info("Voice system cleaned up.")
        except Exception as e:
            logger.error(f"Error during voice system cleanup: {e}", exc_info=True)


    # --- Print Consumer Cleanup ---
    logger.info("Stopping print consumer...")
    try:
        await print_queue.put((None, None)) # Send sentinel
        # Wait briefly for the consumer to process the sentinel
        await asyncio.wait_for(print_task, timeout=2.0)
        logger.info("Print consumer stopped.")
    except asyncio.TimeoutError:
        logger.warning("Print consumer task did not finish promptly. Cancelling.")
        print_task.cancel()
        try: await print_task # Allow cancellation to be processed
        except asyncio.CancelledError: pass # Expected
    except Exception as e:
        logger.error(f"Error stopping print consumer: {e}")

    # Listener thread is daemon, will exit when main thread exits.

# --- Main execution block ---
if __name__ == "__main__":
    try:
        # Ensure terminal is reset properly on exit, especially if errors occur
        import os
        original_stty = None
        if sys.stdin.isatty(): # Check if running in a real terminal
             try:
                 # Use os.read rather than os.popen for potentially better compatibility/security
                 # We need a way to run 'stty -g' and read its output. subprocess is better.
                 stty_process = subprocess.run(['stty', '-g'], capture_output=True, text=True, check=False)
                 if stty_process.returncode == 0:
                      original_stty = stty_process.stdout.strip()
                 else:
                      logger.debug(f"stty -g failed: {stty_process.stderr}")
             except FileNotFoundError:
                  logger.debug("'stty' command not found, cannot save terminal settings.")
             except Exception as e: # Catch other potential errors
                  logger.warning(f"Could not get terminal settings via stty: {e}")
                  original_stty = None

        try:
             asyncio.run(async_main())
        finally:
             # Restore terminal settings if they were saved
             if original_stty and sys.stdin.isatty(): # Check again if it's a tty
                 logger.debug(f"Restoring stty settings: {original_stty}")
                 try:
                     # Use subprocess again for consistency
                     restore_process = subprocess.run(['stty', original_stty], check=False)
                     if restore_process.returncode != 0:
                          logger.warning(f"Failed to restore stty settings: {restore_process.stderr}")
                 except FileNotFoundError:
                      logger.warning("Cannot restore terminal settings: 'stty' not found.")
                 except Exception as e:
                      logger.error(f"Error restoring stty settings: {e}")

    except KeyboardInterrupt:
        # This catches Ctrl+C if it happens *before* the asyncio loop starts or *after* it exits
        logger.info("Application interrupted by user (Ctrl+C outside main loop).")
    except Exception as e:
        # Log critical errors that occur outside the main async loop
        logging.critical(f"Application failed to run: {e}", exc_info=True)
        # Ensure error is printed to stderr if logging isn't fully set up
        print(f"\n[CRITICAL ERROR] Application failed: {e}", file=sys.stderr)
        sys.exit(1) # Exit with error code
    finally:
        # Ensure this message always prints on exit
        print("\nVoice Command CLI exited.")
        # Ensure a clean exit code, especially after KeyboardInterrupt handled gracefully
        sys.exit(0)
