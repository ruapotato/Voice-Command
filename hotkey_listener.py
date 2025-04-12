# hotkey_listener.py
import threading
import asyncio
import logging
import subprocess # Add subprocess import
from pynput import keyboard

# --- Globals for Hotkey State ---
ctrl_pressed = False
alt_pressed = False
recording_key_pressed = False
# --- New global for Ctrl state ---
ctrl_c_combo_pressed = False # Tracks if Ctrl+C is currently held

logger = logging.getLogger(__name__)
logging.getLogger("pynput").setLevel(logging.WARNING) # Quieten pynput

# --- References (set during initialization) ---
voice_system_ref = None
print_scheduler_ref = None
main_loop_ref = None # Reference to the main event loop
# --- New reference to the current task in main.py ---
current_task_accessor = None # Function to get/cancel the current task

# --- Interrupt Function (scheduled on main loop) ---
def _interrupt_current_action():
    """Cancels the current command task and stops speech."""
    if not current_task_accessor or not print_scheduler_ref:
        logger.warning("Cannot interrupt: Missing task accessor or print scheduler.")
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
            logger.debug("Interrupt requested: No active command task to cancel.")
    except Exception as e:
        logger.error(f"Error accessing/cancelling current task: {e}")

    speech_stopped = False
    try:
        # Stop espeak forcefully
        logger.debug("Interrupt requested: Stopping any active espeak process.")
        # Use run instead of Popen as it's simpler for a quick kill
        # Run this synchronously within the scheduled call
        result = subprocess.run(['pkill', '-f', 'espeak'], capture_output=True, check=False)
        if result.returncode == 0:
            logger.debug("pkill stopped espeak process(es).")
            speech_stopped = True
        elif result.returncode == 1:
            logger.debug("pkill found no espeak process to stop.")
        else:
            logger.warning(f"pkill failed for espeak: {result.stderr.decode(errors='ignore')}")
    except FileNotFoundError:
        logger.error("Cannot stop speech: 'pkill' command not found.")
    except Exception as e:
        logger.error(f"Error running pkill for espeak: {e}")

    # Print message only if something was likely stopped or cancelled
    if task_cancelled or speech_stopped:
         # Schedule print back to the main thread context via the scheduler
        if print_scheduler_ref and main_loop_ref:
             main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Interrupted by user (Ctrl+C).")
    else:
         # Maybe print a "Nothing to interrupt" message? Optional.
         pass


# --- Listener Functions ---

def on_press(key):
    global ctrl_pressed, alt_pressed, recording_key_pressed, ctrl_c_combo_pressed
    global voice_system_ref, print_scheduler_ref, main_loop_ref, current_task_accessor # Added current_task_accessor

    # --- Early exit if refs aren't set ---
    if not voice_system_ref or not print_scheduler_ref or not main_loop_ref or not current_task_accessor:
        return

    try:
        # --- Recording Hotkey Logic (Ctrl+Alt) ---
        is_ctrl = key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r
        is_alt = key == keyboard.Key.alt_l or key == keyboard.Key.alt_r

        if is_ctrl:
            ctrl_pressed = True
        elif is_alt:
            alt_pressed = True

        if ctrl_pressed and alt_pressed and not recording_key_pressed:
            logger.debug("Hotkey pressed, scheduling recording start.")
            recording_key_pressed = True # Set state immediately

            # Schedule actions on the main loop
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Recording started...")
            main_loop_ref.call_soon_threadsafe(
                lambda: main_loop_ref.run_in_executor(None, voice_system_ref.start_quick_record)
            )
            # Prevent Ctrl+Alt+C from triggering both recording start and interrupt
            return # Don't process Ctrl+C if recording just started

        # --- Interruption Hotkey Logic (Ctrl+C) ---
        # Check if Ctrl is pressed AND the key is 'c'
        if ctrl_pressed and not is_ctrl and not is_alt: # Ensure it's not the Ctrl key itself
             is_c_key = False
             if hasattr(key, 'char') and key.char == 'c':
                 is_c_key = True
             # Deprecated way for some systems/versions?
             # elif hasattr(key, 'vk') and key.vk == 99: # vk for 'c' is often 67 or 99
             #     is_c_key = True

             if is_c_key and not ctrl_c_combo_pressed:
                 logger.debug("Ctrl+C pressed, scheduling interruption.")
                 ctrl_c_combo_pressed = True # Set flag to prevent rapid repeats if key repeats
                 # Schedule the interruption function
                 main_loop_ref.call_soon_threadsafe(_interrupt_current_action)


    except Exception as e:
        logger.error(f"Hotkey press error: {e}", exc_info=True)
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", f"Hotkey press error: {e}")


def on_release(key):
    global ctrl_pressed, alt_pressed, recording_key_pressed, ctrl_c_combo_pressed
    global voice_system_ref, print_scheduler_ref, main_loop_ref
    should_stop_recording = False # Flag to determine if stop should be scheduled

    if not voice_system_ref or not print_scheduler_ref or not main_loop_ref:
        return

    try:
        is_ctrl = key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r
        is_alt = key == keyboard.Key.alt_l or key == keyboard.Key.alt_r
        is_c_key = False
        if hasattr(key, 'char') and key.char == 'c':
            is_c_key = True

        # --- Recording Hotkey Release Logic ---
        if is_ctrl:
            ctrl_pressed = False
            if recording_key_pressed: # Only check if we were recording
                should_stop_recording = True
        elif is_alt:
            alt_pressed = False
            if recording_key_pressed: # Only check if we were recording
                should_stop_recording = True
        # If any other key is released while recording was active, stop it
        elif recording_key_pressed and not (is_ctrl or is_alt):
             should_stop_recording = True


        # Schedule recording stop if needed
        if should_stop_recording and recording_key_pressed: # Check recording_key_pressed again
            logger.debug("Hotkey released, scheduling recording stop.")
            recording_key_pressed = False # Reset state *before* scheduling

            # Schedule actions on the main loop
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Recording stopped. Processing...")
            main_loop_ref.call_soon_threadsafe(
                lambda: main_loop_ref.run_in_executor(None, voice_system_ref.stop_quick_record)
            )

        # --- Interruption Hotkey Release Logic ---
        if is_ctrl:
            ctrl_pressed = False # Already done above, but safe to repeat
            # If C was also held, release the combo lock
            if ctrl_c_combo_pressed:
                 ctrl_c_combo_pressed = False
        elif is_c_key:
             # If C is released, release the combo lock
             if ctrl_c_combo_pressed:
                 ctrl_c_combo_pressed = False

    except Exception as e:
        logger.error(f"Hotkey release error: {e}", exc_info=True)
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", f"Hotkey release error: {e}")


# --- Modified start_listener ---
def start_listener(loop, voice_system_instance, print_scheduler, task_accessor_func):
    """Initializes and starts the global hotkey listener in a daemon thread."""
    global voice_system_ref, print_scheduler_ref, main_loop_ref, current_task_accessor # Added task_accessor
    voice_system_ref = voice_system_instance
    print_scheduler_ref = print_scheduler
    main_loop_ref = loop # Store the loop reference
    current_task_accessor = task_accessor_func # Store the function

    logger.info("Starting global hotkey listener thread (Ctrl+Alt for record, Ctrl+C for interrupt)...")
    try:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener_thread = threading.Thread(target=listener.run, daemon=True, name="HotkeyListenerThread")
        listener_thread.start()
        logger.info("Hotkey listener thread started.")
        return listener_thread # Return thread if needed elsewhere
    except Exception as e:
        logger.error(f"Failed to start hotkey listener: {e}", exc_info=True)
        # Use call_soon_threadsafe even for reporting startup error
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", "Failed to start global hotkey listener!")
        else:
            print("[ERROR] Failed to start global hotkey listener!")
        return None
