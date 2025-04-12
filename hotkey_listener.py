import threading
import asyncio
import logging
from pynput import keyboard

# --- Globals for Hotkey State ---
ctrl_pressed = False
alt_pressed = False
recording_key_pressed = False

logger = logging.getLogger(__name__)
logging.getLogger("pynput").setLevel(logging.WARNING) # Quieten pynput

# --- References (set during initialization) ---
voice_system_ref = None
print_scheduler_ref = None
main_loop_ref = None # Reference to the main event loop

# --- Listener Functions ---

def on_press(key):
    global ctrl_pressed, alt_pressed, recording_key_pressed
    global voice_system_ref, print_scheduler_ref, main_loop_ref

    if not voice_system_ref or not print_scheduler_ref or not main_loop_ref:
        return

    try:
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            ctrl_pressed = True
        elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            alt_pressed = True

        # Use main_loop_ref.call_soon_threadsafe to ensure thread safety when modifying state
        # and scheduling actions from the pynput thread.
        if ctrl_pressed and alt_pressed and not recording_key_pressed:
            logger.debug("Hotkey pressed, scheduling recording start.")
            # Need to set recording_key_pressed safely if accessed by release handler too
            # For simplicity, let's set it here, assuming quick succession is unlikely to cause race
            recording_key_pressed = True

            # Schedule actions on the main loop
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Recording started...")
            # Schedule the SYNCHRONOUS function call using run_in_executor
            main_loop_ref.call_soon_threadsafe(
                lambda: main_loop_ref.run_in_executor(None, voice_system_ref.start_quick_record)
                # 'None' uses the default ThreadPoolExecutor
                # We pass the function object itself, not call it here.
            )
    except Exception as e:
        logger.error(f"Hotkey press error: {e}", exc_info=True)
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", f"Hotkey press error: {e}")

def on_release(key):
    global ctrl_pressed, alt_pressed, recording_key_pressed
    global voice_system_ref, print_scheduler_ref, main_loop_ref
    should_stop = False # Flag to determine if stop should be scheduled

    if not voice_system_ref or not print_scheduler_ref or not main_loop_ref:
        return

    try:
        key_to_check = key
        # Check which key was released and update state
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            ctrl_pressed = False
            # Check if recording was active and the combo is now broken
            if recording_key_pressed and not alt_pressed:
                should_stop = True
        elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            alt_pressed = False
            # Check if recording was active and the combo is now broken
            if recording_key_pressed and not ctrl_pressed:
                should_stop = True
        elif recording_key_pressed and not (ctrl_pressed and alt_pressed):
            # If another key was released AND the combo was already broken, still trigger stop
            # Or if combo was active but now isn't after state update
             should_stop = True

        # Only schedule stop actions if we determined we should stop AND recording was active
        if should_stop and recording_key_pressed:
            logger.debug("Hotkey released, scheduling recording stop.")
            recording_key_pressed = False # Reset state *before* scheduling potentially slow action

            # Schedule actions on the main loop
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Recording stopped. Processing...")
            # Schedule the SYNCHRONOUS function call using run_in_executor
            main_loop_ref.call_soon_threadsafe(
                 lambda: main_loop_ref.run_in_executor(None, voice_system_ref.stop_quick_record)
            )

    except Exception as e:
        logger.error(f"Hotkey release error: {e}", exc_info=True)
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", f"Hotkey release error: {e}")


def start_listener(loop, voice_system_instance, print_scheduler):
    """Initializes and starts the global hotkey listener in a daemon thread."""
    global voice_system_ref, print_scheduler_ref, main_loop_ref
    voice_system_ref = voice_system_instance
    print_scheduler_ref = print_scheduler
    main_loop_ref = loop # Store the loop reference

    logger.info("Starting global hotkey listener thread (Ctrl+Alt)...")
    try:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener_thread = threading.Thread(target=listener.run, daemon=True, name="HotkeyListenerThread")
        listener_thread.start()
        logger.info("Hotkey listener thread started.")
    except Exception as e:
        logger.error(f"Failed to start hotkey listener: {e}", exc_info=True)
        if print_scheduler_ref and main_loop_ref:
            # Use call_soon_threadsafe even for reporting startup error
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", "Failed to start global hotkey listener!")
        else:
             print("[ERROR] Failed to start global hotkey listener!")
