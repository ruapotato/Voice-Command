#!/usr/bin/env python3

import asyncio
import sys
import logging
from typing import List, Optional, Callable, Any # Add Callable, Any
import textwrap

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
from commands.computer_command import ComputerCommand
from core.voice_system import VoiceCommandSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
ollama_models_list: List[str] = ["mistral"]
computer_command_instance: Optional[ComputerCommand] = None
# --- New global for tracking the current command task ---
current_command_task: Optional[asyncio.Task] = None

# --- Accessor function for the hotkey listener ---
def get_current_task() -> Optional[asyncio.Task]:
    """Returns the currently active command task, if any."""
    global current_command_task
    return current_command_task

# --- Voice System Callback ---
# --- Modified: Needs to handle task creation and cancellation ---
async def handle_voice_command(text: str, command_processor: CommandProcessor):
    """Processes voice commands, manages the task, handles cancellation."""
    global current_command_task
    if current_command_task and not current_command_task.done():
        logger.warning("New voice command received while previous task running. Cancelling previous.")
        current_command_task.cancel()
        try:
            await current_command_task # Allow cancellation to propagate
        except asyncio.CancelledError:
            logger.debug("Previous voice task cancelled successfully.")
        except Exception as e:
            logger.error(f"Error awaiting previous cancelled voice task: {e}")
        finally:
             current_command_task = None # Explicitly clear

    logger.info(f"Processing voice command: {text}")
    # Create a task for the command processing
    current_command_task = asyncio.create_task(
        _execute_command_stream(text, command_processor, "Voice") # Use helper
    )

    try:
        await current_command_task
    except asyncio.CancelledError:
        schedule_print("System", "Voice command processing cancelled.")
        logger.info("Voice command task was cancelled.")
        # Ensure any subprocesses (like espeak) are handled by the cancellation signal in hotkey listener
    except Exception as e:
        logger.error(f"Error executing voice command task '{text}': {e}", exc_info=True)
        schedule_print("Error", f"Failed processing voice command: {e}")
    finally:
        # Ensure the task reference is cleared after completion/cancellation
        if current_command_task and current_command_task.done():
             current_command_task = None


def handle_transcript(text: str, source: str = "Voice"):
    """Callback from Voice System. Schedules print or command processing."""
    # Schedule print for all transcripts initially
    schedule_print(source, text)

    # If it's a final voice transcription (not just "...") , schedule command processing
    # Assuming 'Voice' is the source for final transcriptions needing execution
    # and system/error messages don't trigger commands here.
    if source == "Voice" and text and text != "...":
         # We need access to the command_processor and the loop to schedule the async handler
         # This creates a dependency issue. Let's refactor VoiceCommandSystem slightly.

         # Alternative: VoiceCommandSystem calls a method passed to it during init,
         # which is defined in main.py and has access to command_processor.
         # See modification in VoiceCommandSystem below.
         pass # Command processing will be triggered differently now

# --- Typed Command Processing ---
# --- Modified: Needs to handle task creation and cancellation ---
async def process_typed_command(text: str, command_processor: CommandProcessor):
    """Processes commands entered via the CLI, manages the task, handles cancellation."""
    global current_command_task
    if current_command_task and not current_command_task.done():
        logger.warning("New typed command received while previous task running. Cancelling previous.")
        current_command_task.cancel()
        try:
            await current_command_task # Allow cancellation to propagate
        except asyncio.CancelledError:
            logger.debug("Previous typed task cancelled successfully.")
        except Exception as e:
            logger.error(f"Error awaiting previous cancelled typed task: {e}")
        finally:
            current_command_task = None # Explicitly clear


    schedule_print("Typed", text)
    # Create a task for the command processing
    current_command_task = asyncio.create_task(
        _execute_command_stream(text, command_processor, "Typed") # Use helper
    )

    try:
        await current_command_task
    except asyncio.CancelledError:
        schedule_print("System", "Typed command processing cancelled.")
        logger.info("Typed command task was cancelled.")
        # Ensure any subprocesses (like espeak) are handled by the cancellation signal in hotkey listener
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
    try:
        cmd_name, args = command_processor.parse_command(text)
        processor_input = text # Default to full text
        is_general_query = False

        if cmd_name:
            # For known commands, maybe don't prefix with 'computer'
            # Let the command processor handle the parsed command directly
            schedule_print("System", f"Executing: {cmd_name} {args if args else ''}")
            processor_input = text # Use the original parsed command text
        elif source == "Typed":
            # For typed input that doesn't match a command, treat as general query for ComputerCommand
            schedule_print("System", f"Processing general query: {text}...")
            processor_input = f"computer {text}" # Prepend "computer"
            is_general_query = True
        elif source == "Voice":
             # For voice input that doesn't match, prepend "computer"
             schedule_print("System", f"Processing general voice query: {text}...")
             processor_input = f"computer {text}" # Prepend "computer"
             is_general_query = True


        # Use the potentially modified processor_input
        async for result in command_processor.process_command(processor_input):
            # Determine message type (LLM if general query, System otherwise)
            msg_type = "LLM" if is_general_query else "System"
            schedule_print(msg_type, f"{result}")

            # Speak result conditionally (avoid speaking errors, suggestions, etc.)
            # Let ComputerCommand handle its own speaking for streaming LLM output.
            # Only speak results from other commands explicitly.
            if not is_general_query and result and not isinstance(result, (list, dict)) and \
               not str(result).startswith("[Error:") and \
               not str(result).startswith("Suggested command:") and \
               not str(result).startswith("Attempting action:") and \
               not str(result).startswith("Interrupted"):
                await speak(result)

    except asyncio.CancelledError:
         # Log cancellation specifically for this execution context
         logger.info(f"Command execution for '{text}' cancelled.")
         raise # Re-raise cancellation error to be caught by the caller
    except Exception as e:
        # Log and schedule print for errors during command execution itself
        logger.error(f"Error processing command '{text}' in _execute_command_stream: {e}", exc_info=True)
        schedule_print("Error", f"Failed processing command: {e}")
        # Don't re-raise normal exceptions, let the task finish with error logged.


# --- Help Text Generation (no changes needed) ---
def generate_help_text(command_processor: CommandProcessor) -> str:
    # ... (keep existing implementation) ...
    lines = []; lines.append("Available commands:")
    details = command_processor.get_command_details(); max_len = 0
    if details:
        for name, aliases, _ in details: alias_str = f" ({', '.join(aliases)})" if aliases else ""; max_len = max(max_len, len(name) + len(alias_str))
    static_cmds = ["select model [name]", "refresh_models", "help", "exit / quit", "stop"]
    max_len = max(max_len, max(len(s) for s in static_cmds)) if static_cmds else max_len
    indent = "  "; padding = 2; desc_width = 70
    for name, aliases, description in details:
        alias_str = f" ({', '.join(aliases)})" if aliases else ""; command_part = f"{indent}{name}{alias_str}".ljust(max_len + len(indent) + padding)
        wrapped_desc = textwrap.wrap(description or "No description.", width=desc_width)
        lines.append(f"  <ansiblue>{command_part}</ansiblue> {wrapped_desc[0]}")
        for line in wrapped_desc[1:]: lines.append(f"{indent}{' ' * (max_len + padding)} {line}")
    lines.append("\nOther:")
    lines.append(f"{indent}{'select model [name]'.ljust(max_len + padding)} - Switch the Ollama LLM model.")
    lines.append(f"{indent}{'refresh_models'.ljust(max_len + padding)} - Reload list of available Ollama models.")
    lines.append(f"{indent}{'stop'.ljust(max_len + padding)} - Stops active text-to-speech feedback.")
    lines.append(f"{indent}{'help'.ljust(max_len + padding)} - Shows this help message.")
    lines.append(f"{indent}{'exit / quit'.ljust(max_len + padding)} - Exits the application.")
    lines.append("\nUsage:")
    lines.append(f"{indent}General queries (e.g., 'tell me a joke') can be typed directly.")
    lines.append(f"{indent}Voice activation: Press and hold Ctrl+Alt to record a voice command.")
    lines.append(f"{indent}Interruption: Press Ctrl+C to stop the current command/speech.")
    return "\n".join(lines)


# --- Dynamic Prompt Function (no changes needed) ---
def get_dynamic_prompt() -> str:
    # ... (keep existing implementation) ...
    model = computer_command_instance.llm_model if computer_command_instance else "???"
    return f"{model}> "

# --- Main Application Logic ---
async def async_main():
    """Main asynchronous function for the CLI."""
    global ollama_models_list, computer_command_instance, current_command_task

    main_event_loop = asyncio.get_running_loop()
    print_task = asyncio.create_task(print_consumer())
    # --- Allow print consumer to start ---
    await asyncio.sleep(0.05)

    schedule_print("System", "Initializing Voice Command CLI...")

    command_processor = CommandProcessor()
    # --- Ensure ComputerCommand instance is fetched ---
    computer_command_instance = command_processor.commands.get("computer")
    if not isinstance(computer_command_instance, ComputerCommand):
        schedule_print("Error", "Critical: ComputerCommand module not found. LLM features disabled.")
        # Decide if you want to exit or continue without LLM
        # sys.exit(1) # Or just let it run without LLM
    else:
        # Initialize Ollama Models (check moved inside the 'else')
        schedule_print("System", "Fetching Ollama models...")
        try:
            fetched_models = await computer_command_instance.get_available_models()
            if fetched_models:
                ollama_models_list = fetched_models
                ollama_models_for_completion[:] = ollama_models_list
                # Set default model only if list isn't empty
                if ollama_models_list:
                     computer_command_instance.set_llm_model(ollama_models_list[0])
                     schedule_print("System", f"Ollama models loaded. Using: {computer_command_instance.llm_model}")
                else:
                     schedule_print("Warning", "No Ollama models found. LLM commands may fail.")
                     # Keep default 'mistral' or handle appropriately
                     computer_command_instance.set_llm_model("mistral") # Fallback
            else:
                schedule_print("Warning", f"Could not fetch Ollama models. Using default: {ollama_models_list[0]}")
                ollama_models_for_completion[:] = ollama_models_list # Update completer with default
                computer_command_instance.set_llm_model(ollama_models_list[0]) # Use default
        except Exception as e:
            schedule_print("Error", f"Failed to fetch Ollama models: {e}. Using default.")
            ollama_models_for_completion[:] = ollama_models_list # Update completer with default
            if computer_command_instance: computer_command_instance.set_llm_model(ollama_models_list[0])


    # --- Initialize Voice System & Hotkey ---
    voice_system = None # Define voice_system before try block
    listener_thread = None # Define listener_thread before try block

    # --- Define the callback function that VoiceCommandSystem will call ---
    async def trigger_command_processing(transcribed_text: str):
        # This function now runs in the main async context and has access
        # to command_processor.
        await handle_voice_command(transcribed_text, command_processor)

    try:
        # Pass the new trigger function during initialization
        voice_system = VoiceCommandSystem(
             loop=main_event_loop,
             speak_func=speak,
             command_trigger_func=trigger_command_processing # Pass the new async callback
        )
        # The transcript callback is now only for printing
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
        schedule_print("Error", "Failed to initialize voice system or hotkey. Voice commands disabled.")
        if voice_system: # Attempt cleanup if voice system partially initialized
             try:
                 if asyncio.iscoroutinefunction(voice_system.cleanup): await voice_system.cleanup()
                 else: voice_system.cleanup()
             except Exception as cleanup_e: logger.error(f"Error during voice system cleanup after init failure: {cleanup_e}")
        voice_system = None # Ensure voice_system is None if init fails
        # Do not proceed to CLI loop if basic systems failed? Or allow text-only mode?
        # For now, let it proceed to text-only CLI

    # --- Setup Prompt Session ---
    cli_completer = CLICompleter(command_processor)
    session = PromptSession(
        get_dynamic_prompt, # Dynamic prompt function
        completer=cli_completer,
        complete_while_typing=True,
        # history=FileHistory('cli_history.txt'), # Optional history
        # auto_suggest=AutoSuggestFromHistory(), # Optional suggestions
    )

    schedule_print("System", f"CLI Ready. Type 'help' for commands or use hotkeys.")

    # --- Main Input Loop ---
    while True:
        input_text = "" # Ensure defined in outer scope
        try:
            # Use patch_stdout to ensure prompt redraws correctly after async prints
            with patch_stdout():
                input_text = await session.prompt_async() # Removed default=""

            input_text = input_text.strip()
            if not input_text: continue # Ignore empty input

            # --- Handle Special CLI Commands ---
            if input_text.lower() in ["exit", "quit"]:
                schedule_print("System", "Exiting...")
                break # Exit the main loop

            elif input_text.lower() == "help":
                help_content = generate_help_text(command_processor)
                # Use safe_print directly for potentially long help text
                # Need to ensure safe_print handles ANSI codes correctly if used in help_content
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
                                 # Decide how to handle current model if it's gone
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
                    # Handle case where model list is empty or model not found
                    elif not ollama_models_list:
                         schedule_print("Error", "No models available to select.")
                    else:
                         schedule_print("Error", f"Model '{model_name}' not found. Available: {ollama_models_list}")
                else: schedule_print("Error", "Usage: select model <model_name>")
                continue

            # --- Process Regular Commands / General Queries ---
            await process_typed_command(input_text, command_processor)

        except KeyboardInterrupt:
            # Handle Ctrl+C at the prompt - cancel current task if any, or just redraw prompt
            if current_command_task and not current_command_task.done():
                 logger.debug("Ctrl+C at prompt: Cancelling active task.")
                 current_command_task.cancel()
                 # No need to await here, let the main loop continue
                 # The _interrupt_current_action in hotkey_listener will handle the message
            else:
                 # Schedule print message for Ctrl+C when idle
                 schedule_print("System", "(Ctrl+C at prompt)")
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

    # --- Cleanup ---
    schedule_print("System", "Shutting down...")

    # --- Cancel any lingering task ---
    if current_command_task and not current_command_task.done():
         logger.info("Shutting down: Cancelling active command task.")
         current_command_task.cancel()
         try:
             await current_command_task
         except asyncio.CancelledError: pass # Expected
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

    # Wait for listener thread? Not strictly necessary as it's daemon, but cleaner.
    # if listener_thread and listener_thread.is_alive():
    #     logger.debug("Waiting for hotkey listener thread to exit...")
    #     # Pynput listener doesn't have a clean stop method exposed easily across platforms
    #     # Relying on daemon=True is usually sufficient. Joining might hang.
    #     # listener_thread.join(timeout=1.0) # Attempt join with timeout

# --- Main execution block (no changes needed) ---
if __name__ == "__main__":
    try:
        asyncio.run(async_main())
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
