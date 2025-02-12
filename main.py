#!/usr/bin/env python3

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Adw, GLib

import sys
import logging
import asyncio
import threading
from gui.window import VoiceCommandWindow
from gui.application import VoiceCommandApp
from core.voice_system import VoiceCommandSystem

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_async_loop():
    """Set up the asyncio event loop integrated with GLib"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def run_loop():
        loop.run_forever()
    
    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    return loop

def main():
    """Main entry point"""
    try:
        # Setup asyncio loop
        loop = setup_async_loop()
        
        # Initialize voice command system
        voice_system = VoiceCommandSystem()
        
        # Create and run GTK application
        app = VoiceCommandApp(voice_system)
        
        exit_code = app.run(None)
        
        # Cleanup
        loop.call_soon_threadsafe(loop.stop)
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("System shutdown by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
