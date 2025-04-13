# Voice Command System

Control your computer with a command-line voice interface. Uses Whisper for speech recognition and supports clicking UI elements, typing text, reading text aloud, interacting with a local LLM, screen capture with OCR, and more.

## Features

* Speech recognition using OpenAI's Whisper model.
* Click commands: Find and click text/buttons on screen using OCR.
* Type commands: Type text using keyboard emulation.
* Read commands: Read highlighted text aloud using text-to-speech.
* Computer commands: Interact with your system (run shell commands, manage apps/windows, query about highlighted text) using a local LLM (Ollama).
* Screengrab command: Select a screen area, perform OCR, and copy the extracted text.
* Stop command: Immediately halts any active text-to-speech playback.
* Rolling buffer: Captures audio just before hotkey activation to avoid missed words.
* Hotkey controls: Use keyboard shortcuts to trigger recording and interrupt actions.

## Requirements

### Python Dependencies

Follow these steps to set up the project environment:

```bash
# Clone the repository
git clone https://github.com/ruapotato/Voice-Command
cd Voice-Command

# Create and activate a virtual environment
python3 -m venv pyenv
source ./pyenv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```

### System Dependencies

This application relies on several system tools. Install them using your package manager (example for Debian/Ubuntu):

```bash
sudo apt-get install xdotool espeak xclip tesseract-ocr pkill <screenshot-tool>
```

* **xdotool**: For emulating keyboard input (Type command).
* **espeak**: For text-to-speech (Read command).
* **xclip**: For accessing clipboard and primary selection (Read, Screengrab commands).
* **tesseract-ocr**: For Optical Character Recognition (Click, Screengrab commands).
* **pkill**: For stopping processes (Stop command, Ctrl+C interrupt). Usually part of the procps or procps-ng package.
* **\<screenshot-tool\>**: Choose one screenshot tool compatible with area selection:
  * gnome-screenshot
  * maim (often used with slop for selection)
  * scrot

### Local LLM Setup

This project uses Ollama with a compatible language model (e.g., Mistral, Llama 3) for the computer command. To set up:

* Install Ollama from ollama.com
* Pull your desired model, for example:

```bash
ollama pull mistral
```

## Usage

1. If you're returning to the project, make sure to reactivate your Python environment:
   ```bash
   source ./pyenv/bin/activate
   ```

2. Ensure the Ollama service is running in the background if you intend to use the computer command.

3. Run the main application from your terminal:
   ```bash
   python main.py
   ```

### Input Methods:
* **Voice**: Press and hold Ctrl+Alt to record a voice command. Release to process.
* **Typed**: Type directly into the application's prompt.

**Note**: The application features tab completion for commands and parameters, making it more efficient to use both with keyboard input and voice commands. This helps when working with complex commands or when you're not sure about exact syntax.

### Command Execution:
Start your input (voice or typed) with a command keyword (e.g., click, read, computer, screengrab, stop) to execute that specific command.

### Examples:
* `click OK` - Clicks the first visible "OK" text/button.
* `Hello World!` - Types the text "Hello World!" (no need for "type" prefix).
* `read` - Reads currently highlighted text aloud.
* `computer list files in downloads` - Asks the LLM to generate and run a command to list files.
* `screengrab` - Allows you to select a screen area for OCR.
* `stop` - Stops any ongoing speech (Ctrl+C also does this).
* `select model mistral` - Changes the Ollama model you're using (replace "mistral" with any model you've pulled).

**Default Behavior**: Any input (voice or typed) that does not start with a recognized command keyword will be automatically typed out as if using the "type" command, preserving capitalization and punctuation from the original input.

### Keyboard Controls:
* **Record Voice**: Press and hold Ctrl+Alt.
* **Interrupt/Stop**: Press Ctrl+C to stop/cancel the currently running command or any active speech output (same as the "stop" command).
* **Exit**: Type `exit` or `quit` at the prompt, or press Ctrl+D.

## Project Structure

```
/home/david/Voice-Command/
├── LICENSE
├── README.md
├── audio
│   └── vad.py
├── cli
│   ├── __init__.py
│   ├── completer.py
│   └── output.py
├── commands
│   ├── __init__.py
│   ├── base.py
│   ├── click_command.py
│   ├── command_processor.py
│   ├── computer_command.py
│   ├── read_command.py
│   ├── screengrab_command.py
│   ├── stop_command.py
│   └── type_command.py
├── core
│   ├── __init__.py
│   └── voice_system.py
├── hotkey_listener.py
├── main.py
├── print_project.py
├── requirements.txt
├── speech
│   └── whisper_processor.py
└── tmp.txt
```

## License

GPL3 by David Hamner
