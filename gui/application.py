import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Adw, Gio

from .window import VoiceCommandWindow
from core.voice_system import VoiceCommandSystem

class VoiceCommandApp(Adw.Application):
    def __init__(self, voice_system):
        super().__init__(application_id="org.voice.command",
                        flags=Gio.ApplicationFlags.FLAGS_NONE)
        
        # Store voice system reference
        self.voice_system = voice_system
        
        # Connect signals
        self.connect('activate', self.on_activate)

    def on_activate(self, app):
        """Create and show the main window"""
        win = VoiceCommandWindow(app, self.voice_system)
        win.present()

def main():
    """Main entry point"""
    # Initialize voice command system
    voice_system = VoiceCommandSystem()
    
    # Create and run GTK application
    app = VoiceCommandApp(voice_system)
    return app.run(None)

if __name__ == "__main__":
    main()
