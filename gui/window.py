import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Vte', '3.91')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Vte, Gdk, GLib, Adw, Gio, Pango
import asyncio
import threading
from datetime import datetime
import subprocess
import logging
from pynput import keyboard
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VoiceCommandWindow(Gtk.ApplicationWindow):
    def __init__(self, app, voice_system):
        super().__init__(application=app)
        
        # Window setup
        self.set_title("Voice Command System")
        self.set_default_size(1200, 600)
        
        # State
        self.is_listening = False
        self.key_activation_mode = True
        self.recording_key_pressed = False
        
        # Keyboard state
        self.ctrl_pressed = False
        self.alt_pressed = False
        
        # Build UI before setting up voice system
        self.setup_ui()
        self.setup_keyboard()
        
        # Store reference to voice command system and set callback
        self.voice_system = voice_system
        self.voice_system.set_transcript_callback(self.on_transcript)
        
        # Terminal setup
        self.terminal_content = ""
        self.command_history = []
        self.voice_system = voice_system
        self.voice_system.set_window(self)  # Add this line
        self.voice_system.set_transcript_callback(self.on_transcript)
        
        logger.debug("Window initialized")

    def setup_ui(self):
        """Create the main UI layout"""
        # Main container using Gtk.Paned for split view
        self.paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.set_child(self.paned)
        self.paned.set_position(600)
        
        # Left side: Voice command interface
        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        left_box.set_margin_start(18)
        left_box.set_margin_end(18)
        left_box.set_margin_top(18)
        left_box.set_margin_bottom(18)
        
        # Control buttons
        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        left_box.append(controls)
        
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
        left_box.append(query_box)
        
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
        left_box.append(scroll)
        
        self.history_list = Gtk.ListBox()
        self.history_list.set_selection_mode(Gtk.SelectionMode.NONE)
        scroll.set_child(self.history_list)
        
        # Status bar
        self.status_label = Gtk.Label()
        self.status_label.set_xalign(0)
        left_box.append(self.status_label)
        
        # Add left side to paned
        self.paned.set_start_child(left_box)
        self.paned.set_resize_start_child(True)
        
        # Right side: Terminal interface
        self.setup_terminal()
        
        self.update_ui_state()
        logger.debug("UI setup complete")

    def setup_terminal(self):
        """Setup the terminal interface"""
        # Terminal container
        terminal_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        
        # Terminal header
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_margin_start(6)
        header.set_margin_end(6)
        header.set_margin_top(6)
        header.set_margin_bottom(6)
        
        # Terminal label
        terminal_label = Gtk.Label(label="Terminal")
        terminal_label.set_xalign(0)
        header.append(terminal_label)
        
        terminal_box.append(header)
        
        # Terminal widget
        self.terminal = Vte.Terminal()
        self.terminal.set_size_request(-1, 400)
        self.terminal.set_font(Pango.FontDescription("Monospace 11"))
        self.terminal.set_scrollback_lines(10000)
        self.terminal.set_mouse_autohide(True)
        self.terminal.connect('contents-changed', self.on_terminal_output)
        
        # Terminal scrolled window
        terminal_scroll = Gtk.ScrolledWindow()
        terminal_scroll.set_child(self.terminal)
        terminal_scroll.set_vexpand(True)
        terminal_box.append(terminal_scroll)
        
        # Add terminal side to paned
        self.paned.set_end_child(terminal_box)
        self.paned.set_resize_end_child(True)
        
        # Spawn terminal
        self.spawn_terminal()

    def spawn_terminal(self):
        """Start the terminal process"""
        self.terminal.spawn_async(
            Vte.PtyFlags.DEFAULT,
            os.environ['HOME'],
            ['/bin/bash'],
            [],
            GLib.SpawnFlags.DO_NOT_REAP_CHILD,
            None,
            None,
            -1,
            None,
            None,
        )

    def execute_shell_command(self, command: str):
        """Execute a command in the terminal"""
        try:
            self.write_to_terminal(f"# Executing: {command}")
            self.terminal.feed_child(f"{command}\n".encode())
            return True
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return False

    def write_to_terminal(self, text: str):
        """Write text to the terminal"""
        self.terminal.feed_child(f"{text}\n".encode())

    def on_terminal_output(self, terminal):
        """Handle terminal output changes"""
        content = self.terminal.get_text()[0].strip()
        self.terminal_content = content
        # Add to command history if it's a command
        if content.startswith('$'):
            cmd = content.split('$', 1)[1].strip()
            if cmd and cmd not in self.command_history:
                self.command_history.append(cmd)
                if len(self.command_history) > 10:
                    self.command_history.pop(0)

    def add_history_item(self, command_type, text):
        """Add item to command history with copy button"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Create history entry
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        box.set_margin_start(6)
        box.set_margin_end(6)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        
        # Header with command type, timestamp, and copy button
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        
        # Left side with type and timestamp
        header_left = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        type_label = Gtk.Label()
        type_label.set_markup(f"<b>{command_type}</b>")
        header_left.append(type_label)
        
        time_label = Gtk.Label(label=timestamp)
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b".dim-label { opacity: 0.55; }")
        time_label.get_style_context().add_provider(
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        header_left.append(time_label)
        
        # Make left side expand
        header_left.set_hexpand(True)
        header.append(header_left)
        
        # Copy button
        copy_button = Gtk.Button()
        copy_button.set_icon_name("edit-copy-symbolic")
        copy_button.set_tooltip_text("Copy to clipboard")
        copy_button.connect("clicked", self.on_copy_text, text)
        header.append(copy_button)
        
        box.append(header)
        
        # Command text
        text_label = Gtk.Label(label=text)
        text_label.set_xalign(0)
        text_label.set_wrap(True)
        text_label.set_selectable(True)  # Make text selectable
        box.append(text_label)
        
        # Add to list
        row = Gtk.ListBoxRow()
        row.set_child(box)
        self.history_list.prepend(row)
        self.history_list.show()
        
        logger.debug(f"Added history item: {command_type} - {text}")
        return False  # Required for GLib.idle_add

    def on_copy_text(self, button, text):
        """Copy text to clipboard"""
        clipboard = self.get_clipboard()
        clipboard.set(text)
        
        # Show a brief notification
        self.status_label.set_text("Copied to clipboard!")
        GLib.timeout_add(1500, self.reset_status_label)

    def reset_status_label(self):
        """Reset the status label to its default text"""
        if self.is_listening:
            self.status_label.set_text("Listening for commands...")
        else:
            if self.key_activation_mode:
                self.status_label.set_text("Press Ctrl+Alt to record command")
            else:
                self.status_label.set_text("Click microphone to start listening")
        return False  # Required for GLib.timeout_add

    def setup_keyboard(self):
        """Setup keyboard listener"""
        try:
            def on_press(key):
                try:
                    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                        self.ctrl_pressed = True
                    elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                        self.alt_pressed = True
                    
                    # Check if hotkey combination is pressed (Ctrl+Alt)
                    if self.ctrl_pressed and self.alt_pressed:
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
                    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                        self.ctrl_pressed = False
                    elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                        self.alt_pressed = False
                    
                    # Stop recording if any required key is released
                    if self.recording_key_pressed and not (self.ctrl_pressed and self.alt_pressed):
                        logger.debug("Stopping quick record...")
                        self.recording_key_pressed = False
                        GLib.idle_add(self.status_label.set_text, "Press Ctrl+Alt to record command")
                        GLib.idle_add(self.voice_system.stop_quick_record)
                except Exception as e:
                    logger.error(f"Error in key release handler: {e}", exc_info=True)

            # Start the keyboard listener in a non-blocking thread
            self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.keyboard_listener.start()
            
            logger.debug("Keyboard listener started")
            
        except Exception as e:
            logger.error(f"Failed to setup keyboard: {e}", exc_info=True)

    def on_transcript(self, text, command_type="Voice"):
        """Handle new transcripts from voice system"""
        logger.info(f"New transcript: {text}")
        GLib.idle_add(self.add_history_item, command_type, text)

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
                self.status_label.set_text("Press Ctrl+Alt to record command")
            else:
                self.status_label.set_text("Click microphone to start listening")
        
        logger.debug(f"UI state updated - Listening: {self.is_listening}, Key Mode: {self.key_activation_mode}")
