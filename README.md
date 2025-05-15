# Voice Command System

Control your computer with a command-line voice interface. Uses NVIDIA's Parakeet-TDT model for speech recognition and supports clicking UI elements, typing text, reading text aloud, interacting with a local LLM, screen capture with OCR, and more.

## Features

* Speech recognition using **NVIDIA Parakeet-TDT 0.6B V2 via NeMo toolkit**, providing accurate transcription with punctuation and capitalization.
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
# Clone the repository (if you haven't already)
# git clone https://github.com/ruapotato/Voice-Command
# cd Voice-Command

# Create and activate a virtual environment (if not already done)
python3 -m venv pyenv
source ./pyenv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```

Key Python libraries include:

- nemo_toolkit[asr]: For NVIDIA NeMo and the Parakeet ASR model.
- torch: PyTorch, a dependency for NeMo.
- pyaudio: For audio input/output.
- numpy: For numerical audio data manipulation.
- webrtcvad-wheels: For Voice Activity Detection.
- pynput: For global hotkey listening.
- pyautogui, Pillow, pytesseract: For screen interaction and OCR (Click, Screengrab commands).
- prompt_toolkit: For the interactive CLI.
- soundfile: For handling audio files.
- (Other dependencies as listed in requirements.txt)

Make sure your requirements.txt reflects these, especially nemo_toolkit[asr]>=1.23.0 (or latest) and soundfile. You should remove openai-whisper if it's still listed, as it's no longer used.

### System Dependencies

This application relies on several system tools. Install them using your package manager (example for Debian/Ubuntu):

```bash
sudo apt-get install xdotool espeak xclip tesseract-ocr pkill <screenshot-tool>
```

- xdotool: For emulating keyboard input (Type command).
- espeak: For text-to-speech (Read command).
- xclip: For accessing clipboard and primary selection (Read, Screengrab commands).
- tesseract-ocr: For Optical Character Recognition (Click, Screengrab commands).
- pkill: For stopping processes (Stop command, Ctrl+C interrupt). Usually part of the procps or procps-ng package.
- libsndfile1: Often needed for soundfile to correctly process WAV files. Install if you encounter issues with temporary audio file handling (sudo apt-get install libsndfile1).
- <screenshot-tool>: Choose one screenshot tool compatible with area selection:
  - gnome-screenshot
  - maim (often used with slop for selection)
  - scrot

### Local LLM Setup

This project uses Ollama with a compatible language model (e.g., Mistral, Llama 3) for the computer command. To set up:

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull your desired model, for example:

```bash
ollama pull mistral
```

(The default model in the application is mistral, but you can change this via the select model CLI command).

## Usage

If you're returning to the project, make sure to reactivate your Python environment:

```bash
source ./pyenv/bin/activate
```

Ensure the Ollama service is running in the background if you intend to use the computer command.

Run the main application from your terminal:

```bash
python main.py
```

The first time you run it after switching to Parakeet, NeMo will download the model files, which might take some time and requires an internet connection.

### Input Methods:

- **Voice**: Press and hold Ctrl+Alt to record a voice command. Release to process.
- **Typed**: Type directly into the application's prompt.

Note: The application features tab completion for commands and parameters, making it more efficient to use both with keyboard input and voice commands. This helps when working with complex commands or when you're not sure about exact syntax.

### Command Execution:

Start your input (voice or typed) with a command keyword (e.g., click, read, computer, screengrab, stop) to execute that specific command.

#### Examples:

- `click OK` - Clicks the first visible "OK" text/button.
- `Hello World!` - Types the text "Hello World!" (no need for "type" prefix).
- `read` - Reads currently highlighted text aloud.
- `computer list files in downloads` - Asks the LLM to generate and run a command to list files.
- `screengrab` - Allows you to select a screen area for OCR.
- `stop` - Stops any ongoing speech (Ctrl+C also does this).
- `select model mistral` - Changes the Ollama model you're using (replace "mistral" with any model you've pulled).

**Default Behavior**: Any input (voice or typed) that does not start with a recognized command keyword will be automatically typed out as if using the "type" command, preserving capitalization and punctuation from the original input.

### Keyboard Controls:

- **Record Voice**: Press and hold Ctrl+Alt.
- **Interrupt/Stop**: Press Ctrl+C to stop/cancel the currently running command or any active speech output (same as the "stop" command).
- **Exit**: Type `exit` or `quit` at the prompt, or press Ctrl+D.

## Project Structure

```
/home/david/Voice-Command/
├── LICENSE
├── README.md
├── audio # Contains old VAD related files, consider cleanup or integration
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
│   └── whisper_processor.py # This now contains the ParakeetProcessor
└── tmp.txt 
```

(Project structure might vary slightly based on your latest changes, e.g., if audio/vad.py was removed or repurposed).

## License

GPL3 by David Hamner
