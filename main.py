#!/usr/bin/env python3

import asyncio
import sys
import logging
from typing import List, Optional
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
import hotkey_listener
from commands.command_processor import CommandProcessor
from commands.computer_command import ComputerCommand
from core.voice_system import VoiceCommandSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
ollama_models_list: List[str] = ["mistral"]
computer_command_instance: Optional[ComputerCommand] = None

# --- Voice System Callback ---
def handle_transcript(text: str, source: str = "Voice"):
    schedule_print(source, text)

# --- Typed Command Processing ---
async def process_typed_command(text: str, command_processor: CommandProcessor):
    """Processes commands entered via the CLI and speaks the result."""
    schedule_print("Typed", text)
    try:
        cmd_name, args = command_processor.parse_command(text)
        if cmd_name:
            schedule_print("System", f"Executing: {cmd_name} {args if args else ''}")
            async for result in command_processor.process_command(text):
                schedule_print("System", f"{result}")
                if result and not result.startswith("[Error:") and not result.startswith("Suggested command:"):
                    await speak(result)
        else:
            schedule_print("System", f"Processing general query: {text}...")
            query_for_processor = f"computer {text}"
            async for result in command_processor.process_command(query_for_processor):
                 schedule_print("LLM", f"{result}")
                 # LLM speaking handled by ComputerCommand._handle_text_query
    except Exception as e:
        logger.error(f"Error processing typed command '{text}': {e}", exc_info=True)
        schedule_print("Error", f"Failed to process command: {e}")

# --- Help Text Generation ---
def generate_help_text(command_processor: CommandProcessor) -> str:
    """Generates help text dynamically from registered commands."""
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
    return "\n".join(lines)

# --- Dynamic Prompt Function ---
def get_dynamic_prompt() -> str:
    """Returns the prompt string including the current model."""
    model = computer_command_instance.llm_model if computer_command_instance else "???"
    return f"{model}> "

# --- Main Application Logic ---
async def async_main():
    """Main asynchronous function for the CLI."""
    global ollama_models_list, computer_command_instance

    main_event_loop = asyncio.get_running_loop()
    print_task = asyncio.create_task(print_consumer())
    await asyncio.sleep(0.01)

    schedule_print("System", "Initializing Voice Command CLI...")

    command_processor = CommandProcessor()
    computer_command_instance = command_processor.commands.get("computer")
    voice_system = None

    # --- Initialize Ollama Models ---
    if not isinstance(computer_command_instance, ComputerCommand):
         schedule_print("Error", "Critical: ComputerCommand module not found. LLM features disabled.")
    else:
         schedule_print("System", "Fetching Ollama models...")
         try:
             fetched_models = await computer_command_instance.get_available_models()
             if fetched_models:
                  ollama_models_list = fetched_models; ollama_models_for_completion[:] = ollama_models_list
                  computer_command_instance.set_llm_model(ollama_models_list[0])
                  schedule_print("System", f"Ollama models loaded. Using: {computer_command_instance.llm_model}")
             else:
                  schedule_print("Warning", f"Could not fetch Ollama models. Using default: {ollama_models_list}")
                  ollama_models_for_completion[:] = ollama_models_list
                  computer_command_instance.set_llm_model(ollama_models_list[0])
         except Exception as e:
             schedule_print("Error", f"Failed to fetch Ollama models: {e}. Using default.")
             ollama_models_for_completion[:] = ollama_models_list
             if computer_command_instance: computer_command_instance.set_llm_model(ollama_models_list[0])

    # --- Initialize Voice System & Hotkey ---
    try:
        voice_system = VoiceCommandSystem(loop=main_event_loop, speak_func=speak)
        voice_system.set_transcript_callback(handle_transcript)
        schedule_print("System", "Voice system initialized.")
        # Start listener only if voice system init succeeded
        hotkey_listener.start_listener(main_event_loop, voice_system, schedule_print)
    except Exception as e:
        logger.error(f"Failed to initialize VoiceCommandSystem: {e}", exc_info=True)
        schedule_print("Error", "Failed to initialize voice system. Voice commands/hotkey disabled.")
        voice_system = None # Important: Ensure voice_system is None if init fails

    # --- Setup Prompt Session ---
    cli_completer = CLICompleter(command_processor)
    session = PromptSession(
        get_dynamic_prompt, # Dynamic prompt function
        completer=cli_completer,
        complete_while_typing=True,
    )

    schedule_print("System", f"CLI Ready. Type 'help' for commands or press Ctrl+Alt to speak.")

    # --- Main Input Loop ---
    while True:
        input_text = "" # Ensure defined in outer scope
        try:
            with patch_stdout():
                 input_text = await session.prompt_async(default="")

            input_text = input_text.strip()
            if not input_text: continue # Ignore empty input

            # --- Handle Special CLI Commands ---
            if input_text.lower() in ["exit", "quit"]:
                schedule_print("System", "Exiting...")
                break # Exit the main loop

            elif input_text.lower() == "help":
                help_content = generate_help_text(command_processor)
                await safe_print(help_content) # Use safe_print directly
                continue

            elif input_text.lower() == "refresh_models":
                if computer_command_instance:
                    schedule_print("System", "Refreshing Ollama models...")
                    try:
                        fetched_models = await computer_command_instance.get_available_models()
                        if fetched_models:
                            ollama_models_list = fetched_models; ollama_models_for_completion[:] = ollama_models_list
                            if computer_command_instance.llm_model not in ollama_models_list:
                                 new_model = ollama_models_list[0] if ollama_models_list else "mistral"
                                 computer_command_instance.set_llm_model(new_model)
                                 schedule_print("System", f"Models refreshed: {ollama_models_list}. Current model reset to {new_model}")
                            else: schedule_print("System", f"Models refreshed: {ollama_models_list}")
                        else: schedule_print("Error", "Failed to fetch models (received empty list).")
                    except Exception as e: schedule_print("Error", f"Failed to refresh models: {e}")
                else: schedule_print("Error", "Computer command module unavailable.")
                continue

            elif input_text.lower().startswith("select model "):
                parts = input_text.split(maxsplit=2)
                if len(parts) == 3:
                    model_name = parts[2]
                    if model_name in ollama_models_list:
                        if computer_command_instance: computer_command_instance.set_llm_model(model_name); schedule_print("System", f"LLM model set to: {model_name}")
                        else: schedule_print("Error", "Computer command module unavailable.")
                    else: schedule_print("Error", f"Model '{model_name}' not found. Available: {ollama_models_list}")
                else: schedule_print("Error", "Usage: select model <model_name>")
                continue

            # --- Process Regular Commands / General Queries ---
            await process_typed_command(input_text, command_processor)

        except KeyboardInterrupt:
            # Just continue to redraw prompt on Ctrl+C
            schedule_print("System", "(Ctrl+C pressed)")
            continue
        except EOFError:
            # Exit gracefully on Ctrl+D
            schedule_print("System", "EOF received. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error in main loop processing input '{input_text}': {e}", exc_info=True)
            schedule_print("Error", f"An unexpected error occurred: {e}")
            await asyncio.sleep(0.1) # Prevent rapid error loops

    # --- Cleanup ---
    schedule_print("System", "Shutting down...")

    # --- Voice System Cleanup ---
    if voice_system and hasattr(voice_system, 'cleanup'):
        logger.info("Calling voice system cleanup...")
        # <<< REFORMATTED try/except block >>>
        try:
            if asyncio.iscoroutinefunction(voice_system.cleanup):
                await voice_system.cleanup()
            else:
                voice_system.cleanup()
            logger.info("Voice system cleaned up.")
        except Exception as e:
            logger.error(f"Error during voice system cleanup: {e}")

    # --- Print Consumer Cleanup ---
    logger.info("Stopping print consumer...")
    try:
        # Send sentinel value to stop the consumer task
        await print_queue.put((None, None))
        # Wait for the task to finish processing the sentinel
        await asyncio.wait_for(print_task, timeout=2.0)
        logger.info("Print consumer stopped.")
    except asyncio.TimeoutError:
        logger.warning("Print consumer task did not finish promptly.")
        # Ensure task cancellation happens on a new line
        if not print_task.done():
            print_task.cancel()
    except Exception as e:
        logger.error(f"Error stopping print consumer: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C outside main loop).")
    except Exception as e:
        # Log critical errors that occur outside the main async loop
        logging.critical(f"Application failed to run: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure this message always prints on exit
        print("\nVoice Command CLI exited.")
        # Ensure a clean exit code, especially after KeyboardInterrupt
        sys.exit(0)
