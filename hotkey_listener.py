# hotkey_listener.py
import threading
import asyncio
import logging
import subprocess # For running pkill
from pynput import keyboard
from typing import Callable, Optional, Any


# --- Globals for Hotkey State ---
ctrl_pressed = False
alt_pressed = False
recording_key_pressed = False
# --- New global for Ctrl state ---
ctrl_c_combo_pressed = False # Tracks if Ctrl+C is currently held

logger = logging.getLogger(__name__)
# Reduce pynput's own logging noise
logging.getLogger("pynput").setLevel(logging.WARNING)

# --- References (set during initialization) ---
voice_system_ref = None
print_scheduler_ref = None
main_loop_ref = None # Reference to the main event loop
# --- Reference to the current task accessor in main.py ---
current_task_accessor = None # Function to get/cancel the current task

# --- Interrupt Function (scheduled on main loop) ---
def _interrupt_current_action():
    """Cancels the current command task and stops speech."""
    if not current_task_accessor or not print_scheduler_ref or not main_loop_ref:
        logger.warning("Cannot interrupt: Missing task accessor, print scheduler, or main loop reference.")
        return

    task_cancelled = False
    try:
        # Get the current task using the accessor
        current_task = current_task_accessor()
        if current_task and not current_task.done():
            logger.debug("Interrupt requested: Cancelling current command task.")
            current_task.cancel()
            task_cancelled = True
        else:
            logger.debug("Interrupt requested: No active/cancellable command task found.")
    except Exception as e:
        logger.error(f"Error accessing/cancelling current task: {e}")

    speech_stopped = False
    try:
        # Stop espeak forcefully using pkill
        logger.debug("Interrupt requested: Stopping any active espeak process via pkill.")
        # Use run instead of Popen as it's simpler for a quick kill
        # Run this synchronously within the scheduled call
        result = subprocess.run(['pkill', '-f', 'espeak'], capture_output=True, check=False, timeout=1) # Added timeout
        if result.returncode == 0:
            logger.debug("pkill stopped espeak process(es).")
            speech_stopped = True
        elif result.returncode == 1:
            logger.debug("pkill found no espeak process to stop.")
        else:
            # pkill encountered an error (other than not finding the process)
            stderr_msg = result.stderr.decode(errors='ignore').strip()
            logger.warning(f"pkill command failed for espeak (code {result.returncode}): {stderr_msg}")
    except FileNotFoundError:
        logger.error("Cannot stop speech: 'pkill' command not found. Is 'procps' package installed?")
    except subprocess.TimeoutExpired:
         logger.warning("pkill command timed out while trying to stop espeak.")
    except Exception as e:
        logger.error(f"Error running pkill for espeak: {e}")

    # Print message only if something was likely stopped or cancelled
    # Use call_soon_threadsafe as this function runs scheduled on the main loop
    if task_cancelled or speech_stopped:
         main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Interrupted by user (Ctrl+C).")
    else:
         # Optionally print if nothing was stopped
         # main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "(Ctrl+C pressed, nothing to interrupt)")
         pass


# --- Listener Functions (on_press, on_release) ---

def on_press(key):
    """Handles key press events for hotkeys."""
    global ctrl_pressed, alt_pressed, recording_key_pressed, ctrl_c_combo_pressed
    global voice_system_ref, print_scheduler_ref, main_loop_ref, current_task_accessor

    # --- Early exit if essential refs aren't set ---
    if not all([voice_system_ref, print_scheduler_ref, main_loop_ref, current_task_accessor]):
        # Log this only once maybe, or at debug level, to avoid noise if called early
        # logger.debug("Hotkey listener refs not fully initialized yet.")
        return

    try:
        # --- Identify Key Type ---
        is_ctrl = key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r
        is_alt = key == keyboard.Key.alt_l or key == keyboard.Key.alt_r
        is_c_key = False
        try: # Check for character keys safely
             if key.char == 'c':
                 is_c_key = True
        except AttributeError:
             pass # Key is not a character key (like Shift, Ctrl, Alt, etc.)


        # --- Update Modifier Key State ---
        if is_ctrl:
            ctrl_pressed = True
        elif is_alt:
            alt_pressed = True

        # --- Recording Hotkey Logic (Ctrl+Alt) ---
        # Check if Ctrl+Alt is pressed *now* and we aren't already recording
        if ctrl_pressed and alt_pressed and not recording_key_pressed:
            logger.debug("Ctrl+Alt pressed, scheduling recording start.")
            recording_key_pressed = True # Set state immediately

            # Schedule actions on the main loop using threadsafe calls
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Recording started...")
            # Schedule the SYNCHRONOUS start function in the loop's executor
            main_loop_ref.call_soon_threadsafe(
                lambda: main_loop_ref.run_in_executor(None, voice_system_ref.start_quick_record)
            )
            # Prevent Ctrl+Alt+C from triggering both recording start and interrupt simultaneously
            return # Exit after handling Ctrl+Alt


        # --- Interruption Hotkey Logic (Ctrl+C) ---
        # Check if Ctrl is pressed AND the key is 'c' AND we haven't already triggered the interrupt
        if ctrl_pressed and is_c_key and not ctrl_c_combo_pressed:
             logger.debug("Ctrl+C pressed, scheduling interruption.")
             ctrl_c_combo_pressed = True # Set flag to prevent repeats from key-repeat events

             # Schedule the interruption function on the main loop
             main_loop_ref.call_soon_threadsafe(_interrupt_current_action)
             return # Exit after handling Ctrl+C


    except Exception as e:
        # Log errors occurring within the listener callback itself
        logger.error(f"Error in hotkey on_press callback: {e}", exc_info=True)
        # Safely schedule error printing if possible
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", f"Hotkey press error: {e}")


def on_release(key):
    """Handles key release events for hotkeys."""
    global ctrl_pressed, alt_pressed, recording_key_pressed, ctrl_c_combo_pressed
    global voice_system_ref, print_scheduler_ref, main_loop_ref
    should_stop_recording = False # Flag to determine if stop should be scheduled

    # --- Early exit if essential refs aren't set ---
    if not all([voice_system_ref, print_scheduler_ref, main_loop_ref]):
        return

    try:
        # --- Identify Key Type ---
        is_ctrl = key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r
        is_alt = key == keyboard.Key.alt_l or key == keyboard.Key.alt_r
        is_c_key = False
        try:
             if key.char == 'c':
                 is_c_key = True
        except AttributeError:
             pass

        # --- Recording Hotkey Release Logic ---
        # Check if a modifier key critical to the recording combo was released while recording
        if (is_ctrl or is_alt) and recording_key_pressed:
             should_stop_recording = True

        # Update modifier states *after* checking the recording condition
        if is_ctrl:
            ctrl_pressed = False
        elif is_alt:
            alt_pressed = False

        # Schedule recording stop if needed
        # Ensure recording_key_pressed is checked *again* before scheduling,
        # in case it was reset by another event concurrently (though unlikely here).
        if should_stop_recording and recording_key_pressed:
            logger.debug("Recording hotkey released, scheduling recording stop.")
            recording_key_pressed = False # Reset state *before* scheduling potentially slow action

            # Schedule actions on the main loop
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Recording stopped. Processing...")
            # Schedule the SYNCHRONOUS stop function in the loop's executor
            main_loop_ref.call_soon_threadsafe(
                lambda: main_loop_ref.run_in_executor(None, voice_system_ref.stop_quick_record)
            )

        # --- Interruption Hotkey Release Logic ---
        # Reset the ctrl_c_combo_pressed flag when either Ctrl or C is released
        if (is_ctrl or is_c_key) and ctrl_c_combo_pressed:
             logger.debug(f"Ctrl+C combo key released ({key}). Resetting combo flag.")
             ctrl_c_combo_pressed = False

        # Update ctrl_pressed state again if it was C being released while Ctrl was still held
        # (This ensures ctrl_pressed is accurate if C is released before Ctrl)
        if is_c_key and not is_ctrl:
             # If C was released, the state of ctrl_pressed hasn't changed here,
             # it reflects whether Ctrl is *still* held down.
             pass


    except Exception as e:
        logger.error(f"Error in hotkey on_release callback: {e}", exc_info=True)
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", f"Hotkey release error: {e}")


# --- Function to Start the Listener ---
def start_listener(loop: asyncio.AbstractEventLoop,
                   voice_system_instance: Any, # Use Any or specific type if available
                   print_scheduler: Callable, # Type hint for scheduler
                   task_accessor_func: Callable[[], Optional[asyncio.Task]]) -> Optional[threading.Thread]:
    """
    Initializes and starts the global hotkey listener in a background daemon thread.

    Args:
        loop: The main asyncio event loop.
        voice_system_instance: The instance of the VoiceCommandSystem.
        print_scheduler: The function to schedule messages for printing.
        task_accessor_func: A function that returns the current command asyncio.Task or None.

    Returns:
        The listener thread object if successful, otherwise None.
    """
    global voice_system_ref, print_scheduler_ref, main_loop_ref, current_task_accessor
    voice_system_ref = voice_system_instance
    print_scheduler_ref = print_scheduler
    main_loop_ref = loop # Store the loop reference
    current_task_accessor = task_accessor_func # Store the task accessor function

    logger.info("Starting global hotkey listener thread (Ctrl+Alt for record, Ctrl+C for interrupt)...")
    try:
        # Create the listener instance
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        # Create and start the thread
        listener_thread = threading.Thread(
            target=listener.run, # The method to run in the new thread
            daemon=True,        # Allow Python to exit even if this thread is running
            name="HotkeyListenerThread" # Assign a name for easier debugging
        )
        listener_thread.start()
        logger.info("Hotkey listener thread started successfully.")
        return listener_thread # Return the thread object
    except Exception as e:
        logger.error(f"Failed to start pynput hotkey listener: {e}", exc_info=True)
        # Use call_soon_threadsafe even for reporting startup error from main thread context
        if print_scheduler_ref and main_loop_ref:
            # Schedule the error message back to the main thread's print queue
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", "CRITICAL: Failed to start global hotkey listener!")
        else:
            # Fallback print if scheduler/loop not available
            print("[CRITICAL ERROR] Failed to start global hotkey listener!")
        return None # Indicate failure
