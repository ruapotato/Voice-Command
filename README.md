# Voice Command System

Control your computer with a voice interface. Uses Whisper for speech recognition and supports clicking UI elements, typing text, reading text aloud, and asking questions about highlighted text.

## Features

- Speech recognition using OpenAI's Whisper model
- Click commands: Find and click text/buttons on screen using OCR
- Type commands: Type text using keyboard emulation
- Read commands: Read highlighted text aloud using text-to-speech
- Computer commands: Ask questions about highlighted text using local LLM
- Rolling buffer to capture speech before hotkey activation

## Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### System Dependencies
```bash
sudo apt-get install xdotool espeak xclip tesseract-ocr
```

### Local LLM Setup
This project uses Ollama with any compatible language model for local LLM processing. To set up:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull your desired model:
```bash
ollama pull mistral  # or any other model like llama3, phi, etc.
```

## Usage

1. Ensure Ollama is running in the background
```bash
# Ollama service should be running automatically after installation
```

2. Run the voice command system:
```bash
python main.py
```

3. Available commands:
- "Click [text]" - Clicks text/buttons on screen
- "Type [text]" or "Type in [text]" - Types the specified text
- "Read" - Reads currently highlighted text aloud
- "Computer [query]" - Asks about highlighted text

4. Keyboard Controls:
- Press and hold **Ctrl+Alt** to record a voice command
- Click the microphone icon to toggle continuous listening mode
- Click the stop button to stop any active text-to-speech

## Project Structure
```
voice_command/
├── LICENSE
├── README.md
├── commands
│   ├── __init__.py
│   ├── base.py
│   ├── click_command.py
│   ├── command_processor.py
│   ├── computer_command.py
│   ├── read_command.py
│   └── type_command.py
├── core
│   ├── __init__.py
│   └── voice_system.py
├── gui
│   ├── __init__.py
│   ├── application.py
│   └── window.py
├── main.py
├── print_project.py
├── requirements.txt
└── speech
    └── whisper_processor.py
```

## License
GPL3 by David Hamner
