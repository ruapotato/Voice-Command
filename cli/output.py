import asyncio
from prompt_toolkit import print_formatted_text, HTML

# Use an asyncio Queue for thread-safe printing from callbacks/threads
# This queue will hold tuples of (message_type: str, message: str)
print_queue = asyncio.Queue()

async def safe_print(formatted_message: str):
    """Asynchronously prints pre-formatted HTML messages without disrupting the prompt."""
    # This function will be run by the main asyncio loop
    # Using print_formatted_text ensures it prints above the current prompt line.
    print_formatted_text(HTML(formatted_message))

async def print_consumer():
    """Consumes messages from the print_queue and prints them safely."""
    print("Starting print consumer...") # Debug
    while True:
        message_type, message = await print_queue.get()
        # print(f"Dequeued: {message_type}, {message}") # Debug

        if message_type is None: # Sentinel value for stopping
            print("Print consumer received stop signal.")
            print_queue.task_done()
            break

        prefix = f"[{message_type}]"
        if message_type == "Voice":
             # Teal color for Voice input
            formatted_message = f"<ansiteal>{prefix}</ansiteal> {message}"
        elif message_type == "System":
            # Yellow color for System messages/results
             formatted_message = f"<ansiyellow>{prefix}</ansiyellow> {message}"
        elif message_type == "Error":
            # Red color for Errors
            formatted_message = f"<ansired>{prefix}</ansired> {message}"
        elif message_type == "Help":
            # Blue color for Help
            formatted_message = f"<ansiblue>{prefix}</ansiblue>\n{message}" # Add newline for help text
        else:
            # Default formatting
            formatted_message = f"{prefix} {message}"

        # Use the safe_print function to output correctly
        # Ensure safe_print is run in the main loop's context if needed,
        # but print_formatted_text often handles this if called from async function.
        await safe_print(formatted_message)
        print_queue.task_done()
    print("Print consumer finished.") # Debug


def schedule_print(message_type: str, message: str):
    """Puts a message onto the print queue from any thread."""
    try:
        # Get the running loop to ensure thread safety
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(print_queue.put_nowait, (message_type, message))
    except RuntimeError:
        # Fallback if called before loop is running or from non-async context without loop access
        # This might happen during very early initialization or shutdown
        print(f"[Fallback Print {message_type}] {message}")
