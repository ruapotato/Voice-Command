import asyncio
from prompt_toolkit import print_formatted_text, HTML
import subprocess # For speak utility
import logging # For speak utility

logger = logging.getLogger(__name__)

# Use an asyncio Queue for thread-safe printing from callbacks/threads
print_queue = asyncio.Queue()

# --- Speak Utility (using subprocess) ---
ESPEAK_CONFIG = ["-ven+f3", "-k5", "-s150"] # Voice config

def _raw_speak(text: str):
    """Synchronous function to call espeak."""
    if not text:
        return
    logger.debug(f"Attempting to speak: {text[:50]}...")
    # Basic cleaning for shell safety, though espeak handles most things
    text_to_speak = text.replace('`', '').replace('"', "'").replace(';', '.')
    try:
        # Kill previous espeak instances first? Might prevent overlap.
        subprocess.run(['pkill', '-f', 'espeak'], check=False) # Ignore error if none running
        # Run espeak with timeout
        subprocess.run(['espeak'] + ESPEAK_CONFIG + [text_to_speak], check=True, timeout=20)
    except FileNotFoundError:
        logger.error("espeak command not found. Cannot speak.")
    except subprocess.TimeoutExpired:
        logger.warning(f"espeak command timed out.")
    except Exception as e:
        logger.error(f"Speech failed: {e}")

async def speak(text: str):
    """Asynchronously speaks text using espeak in an executor."""
    # Use run_in_executor to avoid blocking the main event loop
    if text: # Avoid trying to speak empty results
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: _raw_speak(text))
        except RuntimeError:
             logger.error("No running event loop found to schedule speak.")
        except Exception as e:
             logger.error(f"Error scheduling speak: {e}")

# --- Print Queue Logic (remains same) ---
async def safe_print(formatted_message: str):
    """Asynchronously prints pre-formatted HTML messages without disrupting the prompt."""
    print_formatted_text(HTML(formatted_message))

async def print_consumer():
    """Consumes messages from the print_queue and prints them safely."""
    # print("Starting print consumer...") # Debug
    while True:
        message_type, message = await print_queue.get()
        # print(f"Dequeued: {message_type}, {message}") # Debug
        if message_type is None: # Sentinel value for stopping
             # print("Print consumer received stop signal.") # Debug
             print_queue.task_done(); break

        prefix = f"[{message_type}]"
        if message_type == "Voice": formatted_message = f"<ansiteal>{prefix}</ansiteal> {message}"
        elif message_type == "System": formatted_message = f"<ansiyellow>{prefix}</ansiyellow> {message}"
        elif message_type == "LLM": formatted_message = f"<ansimagenta>{prefix}</ansimagenta> {message}" # Color for LLM
        elif message_type == "Error": formatted_message = f"<ansired>{prefix}</ansired> {message}"
        elif message_type == "Help": formatted_message = f"<ansiblue>{prefix}</ansiblue>\n{message}"
        elif message_type == "Typed": formatted_message = f"<ansicyan>{prefix}</ansicyan> {message}" # Color for typed input
        else: formatted_message = f"{prefix} {message}"

        await safe_print(formatted_message)
        print_queue.task_done()
    # print("Print consumer finished.") # Debug

def schedule_print(message_type: str, message: str):
    """Puts a message onto the print queue from any thread."""
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(print_queue.put_nowait, (message_type, message))
    except RuntimeError:
        print(f"[Fallback Print {message_type}] {message}")
