# hotkey_listener.py
import threading
import asyncio
import logging
import subprocess
import time
from pynput import keyboard
from typing import Callable, Optional, Any

# --- Globals for Hotkey State ---
ctrl_pressed = False
shift_pressed = False
recording_key_pressed = False
ctrl_c_combo_pressed = False

logger = logging.getLogger(__name__)
logging.getLogger("pynput").setLevel(logging.WARNING)

# --- References (set during initialization) ---
voice_system_ref = None
print_scheduler_ref = None
main_loop_ref = None
current_task_accessor = None

def _interrupt_current_action():
    """Cancels the current command task and stops speech."""
    if not current_task_accessor or not print_scheduler_ref or not main_loop_ref:
        logger.warning("Cannot interrupt: Missing references.")
        return

    task_cancelled = False
    try:
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
        logger.debug("Interrupt requested: Stopping any active espeak process via pkill.")
        result = subprocess.run(['pkill', '-f', 'espeak'], capture_output=True, check=False, timeout=1)
        if result.returncode == 0:
            logger.debug("pkill stopped espeak process(es).")
            speech_stopped = True
        elif result.returncode == 1:
            logger.debug("pkill found no espeak process to stop.")
        else:
            stderr_msg = result.stderr.decode(errors='ignore').strip()
            logger.warning(f"pkill command failed for espeak (code {result.returncode}): {stderr_msg}")
    except FileNotFoundError:
        logger.error("Cannot stop speech: 'pkill' command not found.")
    except subprocess.TimeoutExpired:
        logger.warning("pkill command timed out while trying to stop espeak.")
    except Exception as e:
        logger.error(f"Error running pkill for espeak: {e}")

    if task_cancelled or speech_stopped:
        main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Interrupted by user (Ctrl+C).")

def on_press(key):
    """Handles key press events for hotkeys."""
    global ctrl_pressed, shift_pressed, recording_key_pressed, ctrl_c_combo_pressed
    if not all([voice_system_ref, print_scheduler_ref, main_loop_ref, current_task_accessor]):
        return

    try:
        is_ctrl = key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
        is_shift = key in (keyboard.Key.shift, keyboard.Key.shift_r)
        is_c_key = hasattr(key, 'char') and key.char == 'c'

        if is_ctrl:
            ctrl_pressed = True
        elif is_shift:
            shift_pressed = True

        # --- Recording Hotkey Logic (Ctrl + Shift) ---
        if ctrl_pressed and shift_pressed and not recording_key_pressed:
            logger.debug("Ctrl+Shift pressed, scheduling recording start.")
            recording_key_pressed = True

            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Recording started...")
            main_loop_ref.call_soon_threadsafe(
                lambda: main_loop_ref.run_in_executor(None, voice_system_ref.start_quick_record)
            )
            return

        # --- Interruption Hotkey Logic (Ctrl+C) ---
        if ctrl_pressed and is_c_key and not ctrl_c_combo_pressed:
            logger.debug("Ctrl+C pressed, scheduling interruption.")
            ctrl_c_combo_pressed = True
            main_loop_ref.call_soon_threadsafe(_interrupt_current_action)
            return

    except Exception as e:
        logger.error(f"Error in hotkey on_press callback: {e}", exc_info=True)
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", f"Hotkey press error: {e}")

def on_release(key):
    """Handles key release events for hotkeys."""
    global ctrl_pressed, shift_pressed, recording_key_pressed, ctrl_c_combo_pressed
    if not all([voice_system_ref, print_scheduler_ref, main_loop_ref]):
        return

    try:
        is_ctrl = key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
        is_shift = key in (keyboard.Key.shift, keyboard.Key.shift_r)
        is_c_key = hasattr(key, 'char') and key.char == 'c'

        # --- CORRECTED RELEASE LOGIC ---
        # First, update the state based on which key was released.
        if is_ctrl:
            ctrl_pressed = False
        elif is_shift:
            shift_pressed = False

        # Now, check if we should stop recording.
        # This only triggers if we WERE recording AND NEITHER Ctrl NOR Shift is still pressed.
        if recording_key_pressed and not ctrl_pressed and not shift_pressed:
            logger.debug("Ctrl+Shift combo fully released, scheduling recording stop.")
            recording_key_pressed = False # Reset state immediately

            # This small delay gives slow applications time to process the key-up event
            time.sleep(0.1)

            # Schedule actions on the main loop
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "System", "Recording stopped. Processing...")
            main_loop_ref.call_soon_threadsafe(
                lambda: main_loop_ref.run_in_executor(None, voice_system_ref.stop_quick_record)
            )
        # --- END CORRECTED LOGIC ---

        # --- Interruption Hotkey Release Logic ---
        if (is_ctrl or is_c_key) and ctrl_c_combo_pressed:
            logger.debug(f"Ctrl+C combo key released ({key}). Resetting combo flag.")
            ctrl_c_combo_pressed = False

    except Exception as e:
        logger.error(f"Error in hotkey on_release callback: {e}", exc_info=True)
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", f"Hotkey release error: {e}")

def start_listener(loop: asyncio.AbstractEventLoop,
                   voice_system_instance: Any,
                   print_scheduler: Callable,
                   task_accessor_func: Callable[[], Optional[asyncio.Task]]) -> Optional[threading.Thread]:
    """Initializes and starts the global hotkey listener."""
    global voice_system_ref, print_scheduler_ref, main_loop_ref, current_task_accessor
    voice_system_ref = voice_system_instance
    print_scheduler_ref = print_scheduler
    main_loop_ref = loop
    current_task_accessor = task_accessor_func

    logger.info("Starting global hotkey listener thread (Ctrl+Shift for record, Ctrl+C for interrupt)...")
    try:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener_thread = threading.Thread(
            target=listener.run,
            daemon=True,
            name="HotkeyListenerThread"
        )
        listener_thread.start()
        logger.info("Hotkey listener thread started successfully.")
        return listener_thread
    except Exception as e:
        logger.error(f"Failed to start pynput hotkey listener: {e}", exc_info=True)
        if print_scheduler_ref and main_loop_ref:
            main_loop_ref.call_soon_threadsafe(print_scheduler_ref, "Error", "CRITICAL: Failed to start global hotkey listener!")
        else:
            print("[CRITICAL ERROR] Failed to start global hotkey listener!")
        return None
