import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Adw, GLib, Gdk, Gio
import asyncio
import threading
from datetime import datetime
import subprocess
import logging
from pynput import keyboard

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VoiceCommandWindow(Gtk.ApplicationWindow):
    def __init__(self, app, voice_system):
        super().__init__(application=app)
        
        # Window setup
        self.set_title("Voice Command System")
        self.set_default_size(500, 400)
        
        # State
        self.is_listening = False
        self.key_activation_mode = True
        self.recording_key_pressed = False
        
        # Keyboard state
        self.ctrl_pressed = False
        self.alt_pressed = False
        self.space_pressed = False
        
        # Build UI before setting up voice system
        self.setup_ui()
        self.setup_keyboard()
        
        # Store reference to voice command system and set callback
        self.voice_system = voice_system
        self.voice_system.set_transcript_callback(self.on_transcript)
        
        logger.debug("Window initialized")

    def setup_keyboard(self):
        """Setup keyboard listener"""
        try:
            def on_press(key):
                try:
                    logger.debug(f"Key pressed: {key}")
                    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                        self.ctrl_pressed = True
                    elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                        self.alt_pressed = True
                    elif key == keyboard.Key.space:
                        self.space_pressed = True
                    
                    # Check if hotkey combination is pressed
                    if self.ctrl_pressed and self.alt_pressed and self.space_pressed:
                        if not self.is_listening and not self.recording_key_pressed:
                            logger.debug("Starting quick record...")
                            self.recording_key_pressed = True
                            GLib.idle_add(self.status_label.set_text, "Recording command...")
                            GLib.idle_add(self.voice_system.start_quick_record)
                            GLib.idle_add(self.present)
                except Exception as e:
                    logger.error(f"Error in key press handler: {e}", exc_info=True)

            def on_release(key):
                try:
                    logger.debug(f"Key released: {key}")
                    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                        self.ctrl_pressed = False
                    elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                        self.alt_pressed = False
                    elif key == keyboard.Key.space:
                        self.space_pressed = False
                    
                    # Stop recording if any required key is released
                    if self.recording_key_pressed and not (self.ctrl_pressed and self.alt_pressed and self.space_pressed):
                        logger.debug("Stopping quick record...")
                        self.recording_key_pressed = False
                        GLib.idle_add(self.status_label.set_text, "Press Ctrl+Alt+Space to record command")
                        GLib.idle_add(self.voice_system.stop_quick_record)
                except Exception as e:
                    logger.error(f"Error in key release handler: {e}", exc_info=True)

            # Start the keyboard listener in a non-blocking thread
            self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.keyboard_listener.start()
            
            logger.debug("Keyboard listener started")
            
        except Exception as e:
            logger.error(f"Failed to setup keyboard: {e}", exc_info=True)

    def setup_ui(self):
        """Create the main UI layout"""
        # Main container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_start(18)
        main_box.set_margin_end(18)
        main_box.set_margin_top(18)
        main_box.set_margin_bottom(18)
        self.set_child(main_box)
        
        # Control buttons
        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_box.append(controls)
        
        # Listen button
        self.listen_button = Gtk.Button()
        self.listen_button.set_icon_name("audio-input-microphone-symbolic")
        self.listen_button.connect("clicked", self.on_listen_toggled)
        controls.append(self.listen_button)
        
        # Stop readback button
        stop_button = Gtk.Button()
        stop_button.set_icon_name("media-playback-stop-symbolic")
        stop_button.set_tooltip_text("Stop Text-to-Speech")
        stop_button.connect("clicked", self.on_stop_readback)
        controls.append(stop_button)
        
        # LLM query box
        query_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_box.append(query_box)
        
        self.query_entry = Gtk.Entry()
        self.query_entry.set_placeholder_text("Ask about highlighted text...")
        self.query_entry.connect("activate", self.on_query_submit)
        query_box.append(self.query_entry)
        
        send_button = Gtk.Button(label="Send")
        send_button.connect("clicked", self.on_query_submit)
        query_box.append(send_button)
        
        # Command history
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        main_box.append(scroll)
        
        self.history_list = Gtk.ListBox()
        self.history_list.set_selection_mode(Gtk.SelectionMode.NONE)
        scroll.set_child(self.history_list)
        
        # Status bar
        self.status_label = Gtk.Label()
        self.status_label.set_xalign(0)
        main_box.append(self.status_label)
        
        self.update_ui_state()
        logger.debug("UI setup complete")

    def on_transcript(self, text, command_type="Voice"):
        """Handle new transcripts from voice system"""
        logger.info(f"New transcript: {text}")
        GLib.idle_add(self.add_history_item, command_type, text)

    def add_history_item(self, command_type, text):
        """Add item to command history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Create history entry
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        box.set_margin_start(6)
        box.set_margin_end(6)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        
        # Command type and timestamp
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        type_label = Gtk.Label()
        type_label.set_markup(f"<b>{command_type}</b>")
        header.append(type_label)
        
        time_label = Gtk.Label(label=timestamp)
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b".dim-label { opacity: 0.55; }")
        time_label.get_style_context().add_provider(
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        header.append(time_label)
        
        box.append(header)
        
        # Command text
        text_label = Gtk.Label(label=text)
        text_label.set_xalign(0)
        text_label.set_wrap(True)
        box.append(text_label)
        
        # Add to list
        row = Gtk.ListBoxRow()
        row.set_child(box)
        self.history_list.prepend(row)
        self.history_list.show()
        
        logger.debug(f"Added history item: {command_type} - {text}")
        return False  # Required for GLib.idle_add

    def on_listen_toggled(self, button):
        """Toggle continuous listening mode"""
        self.is_listening = not self.is_listening
        if self.is_listening:
            logger.info("Starting continuous listening")
            self.key_activation_mode = False
            self.voice_system.start_listening()
        else:
            logger.info("Stopping continuous listening")
            self.voice_system.stop_listening()
        self.update_ui_state()

    def on_stop_readback(self, button):
        """Stop any active text-to-speech"""
        logger.info("Stopping text-to-speech")
        try:
            subprocess.run(['pkill', '-9', 'espeak'])
        except Exception as e:
            logger.error(f"Error stopping espeak: {e}", exc_info=True)

    def on_query_submit(self, widget):
        """Submit text query to LLM"""
        query = self.query_entry.get_text().strip()
        if not query:
            return
            
        logger.info(f"Processing text query: {query}")
        self.add_history_item("Query", query)
        
        def run_command():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.voice_system.process_command(f"computer {query}"))
            loop.close()
        
        # Run command in thread
        thread = threading.Thread(target=run_command)
        thread.daemon = True
        thread.start()
        
        self.query_entry.set_text("")

    def update_ui_state(self):
        """Update UI elements based on current state"""
        if self.is_listening:
            self.listen_button.set_icon_name("media-playback-stop-symbolic")
            self.listen_button.set_tooltip_text("Stop Listening")
            self.status_label.set_text("Listening for commands...")
        else:
            self.listen_button.set_icon_name("audio-input-microphone-symbolic")
            self.listen_button.set_tooltip_text("Start Listening")
            if self.key_activation_mode:
                self.status_label.set_text("Press Ctrl+Alt+Space to record command")
            else:
                self.status_label.set_text("Click microphone to start listening")
        
        logger.debug(f"UI state updated - Listening: {self.is_listening}, Key Mode: {self.key_activation_mode}")
