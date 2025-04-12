# cli/output.py
import asyncio
from prompt_toolkit import print_formatted_text, HTML
import subprocess
import logging
import re # Keep for potential future use? Or remove.

logger = logging.getLogger(__name__)

print_queue = asyncio.Queue()
ESPEAK_CONFIG = ["-ven+f3", "-k5", "-s150"] # Voice config

# --- Speak Utility (using subprocess) ---

def _raw_speak(text: str):
    """Synchronous function to call espeak."""
    # Basic check for empty text
    if not text:
        return
    logger.debug(f"Attempting to speak: {text[:50]}...")
    # Basic cleaning for shell safety
    text_to_speak = text.replace('`', '').replace('"', "'").replace(';', '.')
    if not text_to_speak: return # Check again after cleaning

    try:
        # Consider removing automatic pkill if it causes issues with intentional overlaps
        subprocess.run(['pkill', '-f', 'espeak'], check=False, capture_output=True) # Ignore error if none running
        subprocess.run(['espeak'] + ESPEAK_CONFIG + [text_to_speak], check=True, timeout=20)
    except FileNotFoundError:
        logger.error("espeak command not found. Cannot speak.")
    except subprocess.TimeoutExpired:
        logger.warning(f"espeak command timed out for: {text_to_speak[:50]}...")
    except Exception as e:
        logger.error(f"Speech failed for '{text_to_speak[:50]}...': {e}")

async def speak(text: str):
    """
    Asynchronously speaks text using espeak in an executor.
    The caller is responsible for deciding IF text should be spoken.
    """
    # <<< REMOVED _should_speak CHECK >>>
    if text: # Only proceed if text is not empty
        try:
            loop = asyncio.get_running_loop()
            # Run the synchronous _raw_speak in the executor
            await loop.run_in_executor(None, lambda: _raw_speak(text))
        except RuntimeError:
             logger.error("No running event loop found to schedule speak.")
        except Exception as e:
             logger.error(f"Error scheduling speak for '{text[:50]}...': {e}")

# --- Print Queue Logic (remains same) ---
async def safe_print(formatted_message: str):
    """Asynchronously prints pre-formatted HTML messages without disrupting the prompt."""
    print_formatted_text(HTML(formatted_message))

async def print_consumer():
    """Consumes messages from the print_queue and prints them safely."""
    while True:
        message_type, message = await print_queue.get()
        if message_type is None: print_queue.task_done(); break # Sentinel

        prefix = f"[{message_type}]"
        # Apply colors based on type
        if message_type == "Voice": formatted_message = f"<ansiteal>{prefix}</ansiteal> {message}"
        elif message_type == "System": formatted_message = f"<ansiyellow>{prefix}</ansiyellow> {message}"
        elif message_type == "LLM": formatted_message = f"<ansimagenta>{prefix}</ansimagenta> {message}"
        elif message_type == "Error": formatted_message = f"<ansired>{prefix}</ansired> {message}"
        elif message_type == "Help": formatted_message = f"<ansiblue>{prefix}</ansiblue>\n{message}"
        elif message_type == "Typed": formatted_message = f"<ansicyan>{prefix}</ansicyan> {message}"
        else: formatted_message = f"{prefix} {message}" # Default

        await safe_print(formatted_message)
        print_queue.task_done()

def schedule_print(message_type: str, message: str):
    """Puts a message onto the print queue from any thread."""
    # Ensure message is a string
    message_str = str(message) if message is not None else ""
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(print_queue.put_nowait, (message_type, message_str))
    except RuntimeError:
        # Fallback if called before loop is running or from non-async context without loop access
        print(f"[Fallback Print {message_type}] {message_str}")
