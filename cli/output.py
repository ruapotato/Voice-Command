import asyncio
from prompt_toolkit import print_formatted_text, HTML
import subprocess
import logging
import re # For pattern matching

logger = logging.getLogger(__name__)

print_queue = asyncio.Queue()
ESPEAK_CONFIG = ["-ven+f3", "-k5", "-s150"]

# --- List of patterns/prefixes NOT to speak ---
# Use lowercase for case-insensitive matching
DO_NOT_SPEAK_PATTERNS = [
    "[error:",
    "error:",
    "failed",
    "could not",
    "cannot",
    "unknown command",
    "usage:",
    "traceback",
    "critical:",
    "warning:",
    # Add potentially noisy status messages if desired:
    # "executing:",
    # "processing...",
    # "suggested command:",
    # "return code:",
]

def _should_speak(text: str) -> bool:
    """Checks if the given text should be spoken based on patterns."""
    if not text:
        return False
    lower_text = text.lower().strip()
    for pattern in DO_NOT_SPEAK_PATTERNS:
        if lower_text.startswith(pattern):
            logger.debug(f"Skipping speech for text starting with '{pattern}': {text[:50]}...")
            return False
    # Add more complex regex checks if needed
    return True

def _raw_speak(text: str):
    """Synchronous function to call espeak."""
    # Check moved to async speak function
    # if not _should_speak(text): return

    logger.debug(f"Attempting to speak: {text[:50]}...")
    text_to_speak = text.replace('`', '').replace('"', "'").replace(';', '.')
    if not text_to_speak: return
    try:
        # Kill previous espeak to prevent overlap - might be too aggressive?
        # Consider only killing if a new _speak is called quickly after previous one?
        # For now, keep the simple pkill.
        subprocess.run(['pkill', '-f', 'espeak'], check=False, capture_output=True)
        subprocess.run(['espeak'] + ESPEAK_CONFIG + [text_to_speak], check=True, timeout=20)
    except FileNotFoundError: logger.error("espeak command not found. Cannot speak.")
    except subprocess.TimeoutExpired: logger.warning(f"espeak command timed out.")
    except Exception as e: logger.error(f"Speech failed: {e}")

async def speak(text: str):
    """Asynchronously speaks text using espeak in an executor if it should be spoken."""
    # <<< ADDED CHECK HERE >>>
    if text and _should_speak(text):
        try:
            loop = asyncio.get_running_loop()
            # Run the synchronous _raw_speak in the executor
            await loop.run_in_executor(None, lambda: _raw_speak(text))
        except RuntimeError: logger.error("No running event loop found to schedule speak.")
        except Exception as e: logger.error(f"Error scheduling speak: {e}")
    # If _should_speak is false, do nothing

# --- Print Queue Logic (remains same) ---
async def safe_print(formatted_message: str):
    print_formatted_text(HTML(formatted_message))

async def print_consumer():
    # print("Starting print consumer...") # Debug
    while True:
        message_type, message = await print_queue.get()
        # print(f"Dequeued: {message_type}, {message}") # Debug
        if message_type is None: print_queue.task_done(); break # Sentinel

        prefix = f"[{message_type}]"
        if message_type == "Voice": formatted_message = f"<ansiteal>{prefix}</ansiteal> {message}"
        elif message_type == "System": formatted_message = f"<ansiyellow>{prefix}</ansiyellow> {message}"
        elif message_type == "LLM": formatted_message = f"<ansimagenta>{prefix}</ansimagenta> {message}"
        elif message_type == "Error": formatted_message = f"<ansired>{prefix}</ansired> {message}"
        elif message_type == "Help": formatted_message = f"<ansiblue>{prefix}</ansiblue>\n{message}"
        elif message_type == "Typed": formatted_message = f"<ansicyan>{prefix}</ansicyan> {message}"
        else: formatted_message = f"{prefix} {message}"

        await safe_print(formatted_message)
        print_queue.task_done()
    # print("Print consumer finished.") # Debug

def schedule_print(message_type: str, message: str):
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(print_queue.put_nowait, (message_type, message))
    except RuntimeError:
        print(f"[Fallback Print {message_type}] {message}")
