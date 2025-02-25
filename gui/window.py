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
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VoiceCommandWindow(Gtk.ApplicationWindow):
    def __init__(self, app, voice_system):
        super().__init__(application=app)
        
        # Window setup
        self.set_title("Voice Command")
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
        self.voice_system.set_window(self)
        
        # Initialize model list
        self.refresh_models()
        
        # Connect model selection signal
        self.model_dropdown.connect("notify::selected", self.on_model_changed)
        
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
        
        # Listen button GOING AWAY
        #self.listen_button = Gtk.Button()
        #self.listen_button.set_icon_name("audio-input-microphone-symbolic")
        #self.listen_button.connect("clicked", self.on_listen_toggled)
        #controls.append(self.listen_button)
        
        # Stop readback button
        stop_button = Gtk.Button()
        stop_button.set_icon_name("media-playback-stop-symbolic")
        stop_button.set_tooltip_text("Stop Text-to-Speech")
        stop_button.connect("clicked", self.on_stop_readback)
        controls.append(stop_button)
        
        # Add model selection dropdown - NEW CODE
        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        model_box.set_margin_top(6)
        left_box.append(model_box)
        
        model_label = Gtk.Label(label="LLM Model:")
        model_box.append(model_label)
        
        self.model_dropdown = Gtk.DropDown()
        self.model_dropdown.set_margin_start(10)
        self.model_dropdown.set_hexpand(True)
        
        # Populate models (we'll update this list when we detect installed models)
        self.model_factory = Gtk.SignalListItemFactory()
        self.model_factory.connect("setup", self._setup_model_item)
        self.model_factory.connect("bind", self._bind_model_item)
        
        self.model_dropdown.set_factory(self.model_factory)
        model_box.append(self.model_dropdown)
        
        # Refresh button for models
        refresh_button = Gtk.Button()
        refresh_button.set_icon_name("view-refresh-symbolic")
        refresh_button.set_tooltip_text("Refresh Models")
        refresh_button.connect("clicked", self.on_refresh_models)
        model_box.append(refresh_button)
        
        # LLM query box
        query_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        left_box.append(query_box)
        
         # Add help expander before the command history
        help_expander = Gtk.Expander(label="Command Help")
        help_expander.set_expanded(True)
        help_expander.set_margin_top(12)
        
        # Help content in a scrolled window
        help_scroll = Gtk.ScrolledWindow()
        help_scroll.set_min_content_height(150)
        help_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        help_box.set_margin_start(8)
        help_box.set_margin_end(8)
        help_box.set_margin_top(8)
        help_box.set_margin_bottom(8)
        
        # Basic usage section
        basic_label = Gtk.Label()
        basic_label.set_markup("""<span size="large" weight="bold">Basic Usage:</span>
    Press and hold <b>Ctrl + Alt</b> and say a command
    Use the stop button to stop any reply""")
        basic_label.set_xalign(0)
        basic_label.set_wrap(True)
        help_box.append(basic_label)
        
        # Direct commands section
        direct_label = Gtk.Label()
        direct_label.set_markup("""
    <span size="large" weight="bold">Direct Commands:</span>
    <i>These commands don't need LLM processing:</i>

    <span foreground="#2563eb">type</span> <span foreground="#16a34a">"message to type"</span> - Types the specified text
    <span foreground="#2563eb">read</span> - Reads highlighted text
    <span foreground="#2563eb">click</span> <span foreground="#16a34a">"element"</span> - Clicks specified element on screen""")
        direct_label.set_xalign(0)
        direct_label.set_wrap(True)
        direct_label.set_margin_top(12)
        help_box.append(direct_label)
        
        # LLM commands section
        llm_label = Gtk.Label()
        llm_label.set_markup("""
    <span size="large" weight="bold">LLM Commands:</span>
    <i>All commands starting with "computer" use LLM processing:</i>

    <span foreground="#2563eb">computer</span> <span foreground="#16a34a">"question about highlighted text"</span>
    <span foreground="#2563eb">computer</span> <span foreground="#9333ea">shell</span> <span foreground="#16a34a">"request"</span> - Executes bash commands
    <span foreground="#2563eb">computer</span> <span foreground="#9333ea">open</span> <span foreground="#16a34a">"app"</span> - Opens specified application
    <span foreground="#2563eb">computer</span> <span foreground="#9333ea">goto</span> <span foreground="#16a34a">"app"</span> - Focuses specified window
    <span foreground="#2563eb">computer</span> <span foreground="#9333ea">close</span> <span foreground="#16a34a">"app"</span> - Closes specified application""")
        llm_label.set_xalign(0)
        llm_label.set_wrap(True)
        llm_label.set_margin_top(12)
        help_box.append(llm_label)
        
        # Examples section
        examples_label = Gtk.Label()
        examples_label.set_markup("""
    <span size="large" weight="bold">Example Uses:</span>

    - Select a word and say: <span foreground="#2563eb">computer</span> <span foreground="#9333ea">define</span>
    - Select a webpage and say: <span foreground="#2563eb">computer</span> <span foreground="#9333ea">tldr</span>
    - Select a text box and say: <span foreground="#2563eb">type</span> <span foreground="#16a34a">"some really long thing I don't want to type"</span>
    - Select some text and say: <span foreground="#2563eb">read</span>""")
        examples_label.set_xalign(0)
        examples_label.set_wrap(True)
        examples_label.set_margin_top(12)
        help_box.append(examples_label)
        
        # Color guide
        color_label = Gtk.Label()
        color_label.set_markup("""
    <span size="small" weight="bold">Color Guide:</span>
    <span foreground="#2563eb">Blue</span> - Base commands
    <span foreground="#9333ea">Purple</span> - Command tools/modifiers
    <span foreground="#16a34a">Green</span> - Command arguments""")
        color_label.set_xalign(0)
        color_label.set_wrap(True)
        color_label.set_margin_top(12)
        help_box.append(color_label)
        
        help_scroll.set_child(help_box)
        help_expander.set_child(help_scroll)
        left_box.append(help_expander)
        
        # Command history (keep existing code)
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        left_box.append(scroll)
        
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

    def _setup_model_item(self, factory, list_item):
        """Set up model dropdown item"""
        label = Gtk.Label()
        label.set_halign(Gtk.Align.START)
        list_item.set_child(label)

    def _bind_model_item(self, factory, list_item):
        """Bind model dropdown item"""
        label = list_item.get_child()
        item = list_item.get_item()
        
        if item:
            model_name = item.get_string()
            label.set_text(model_name)
    
    def on_model_changed(self, dropdown, param):
        """Handle model selection change"""
        self.update_selected_model()

    def refresh_models(self):
        """Refresh the list of available models"""
        try:
            # Query Ollama for available models
            response = requests.get("http://localhost:11434/api/tags")
            
            if response.ok:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                
                # Create a string list
                model_store = Gtk.StringList()
                for model_name in sorted(model_names):
                    model_store.append(model_name)
                
                # If we have models
                if len(model_names) > 0:
                    self.model_dropdown.set_model(model_store)
                    
                    # Select the first model or try to find "mistral"
                    mistral_index = -1
                    for i, name in enumerate(model_names):
                        if name.lower() == "mistral":
                            mistral_index = i
                            break
                    
                    if mistral_index >= 0:
                        self.model_dropdown.set_selected(mistral_index)
                    else:
                        self.model_dropdown.set_selected(0)
                    
                    # Update the computer command with the selected model
                    self.update_selected_model()
                
                return len(model_names) > 0
                
            else:
                logger.error(f"Failed to get models: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error refreshing models: {e}")
            
            # Create a fallback model list with just "mistral"
            model_store = Gtk.StringList()
            model_store.append("mistral")
            self.model_dropdown.set_model(model_store)
            self.model_dropdown.set_selected(0)
            return False

    def on_refresh_models(self, button):
        """Handle refresh models button click"""
        success = self.refresh_models()
        if success:
            self.status_label.set_text("Models refreshed")
        else:
            self.status_label.set_text("Failed to refresh models. Is Ollama running?")
        GLib.timeout_add(2000, self.reset_status_label)

    def update_selected_model(self):
        """Update the selected model in the ComputerCommand"""
        selected = self.model_dropdown.get_selected()
        model_store = self.model_dropdown.get_model()
        
        if selected >= 0 and model_store:
            model_name = model_store.get_string(selected)
            
            # Update the model in the command processor
            for cmd_name, command in self.voice_system.command_processor.commands.items():
                if cmd_name == "computer" and hasattr(command, 'set_llm_model'):
                    command.set_llm_model(model_name)
                    self.status_label.set_text(f"Model set to {model_name}")
                    GLib.timeout_add(2000, self.reset_status_label)
                    break



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
            #self.listen_button.set_icon_name("media-playback-stop-symbolic")
            #self.listen_button.set_tooltip_text("Stop Listening")
            self.status_label.set_text("Listening for commands...")
        else:
            #self.listen_button.set_icon_name("audio-input-microphone-symbolic")
            #self.listen_button.set_tooltip_text("Start Listening")
            if self.key_activation_mode:
                self.status_label.set_text("Press Ctrl+Alt to record command")
            else:
                self.status_label.set_text("Click microphone to start listening")
        
        logger.debug(f"UI state updated - Listening: {self.is_listening}, Key Mode: {self.key_activation_mode}")
