# Voice Command System

Control your computer with a voice interface. Uses Whisper for speech recognition and supports clicking UI elements, typing text, and asking questions about highlighted text.

## Features
- Voice Activity Detection (VAD) for automatic command detection
- Speech recognition using OpenAI's Whisper model
- Click commands: Find and click text/buttons on screen using OCR
- Type commands: Type text using keyboard emulation
- Computer commands: Ask questions about highlighted text using local LLM

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
This project uses Ollama with the Mistral model for local LLM processing. To set up:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Mistral model:
```bash
ollama pull mistral
```

## Usage

1. Start Ollama:
```bash
ollama run mistral
```

2. Run the voice command system:
```bash
python main.py
```

3. Available commands:
- "Click [text]" - Clicks text/buttons on screen
- "Type [text]" - Types the specified text
- "Computer [query]" - Asks about highlighted text

## Project Structure
```
voice_command/
├── audio/
│   └── vad.py
├── commands/
│   └── command_processor.py
├── speech/
│   └── whisper_processor.py
├── main.py
└── requirements.txt
```

## License
GPL3 by David Hamner
