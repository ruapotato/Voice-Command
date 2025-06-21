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

## Installation on openSUSE Tumbleweed

Follow these steps precisely to set up the project environment.

### Step 1: Install System Dependencies

First, install `pyenv` for managing Python versions. Follow the official `pyenv` installation instructions. After that, install the necessary system packages for both building Python and running the application using `zypper`:

```bash
sudo zypper install git-core gcc automake make zlib-devel libbz2-devel libopenssl-devel readline-devel sqlite3-devel xz-devel libffi-devel tk-devel xdotool espeak xclip tesseract-ocr pkill wmctrl ffmpeg gnome-screenshot
```

### Step 2: Install Correct Python Version

The heavy dependencies like nemo_toolkit require a specific Python version for which pre-compiled packages (wheels) are available. We will use pyenv to install Python 3.11.

```bash
# Install Python 3.11.10 (or latest 3.11.x)
pyenv install 3.11.10

# Create a dedicated virtual environment for the project
pyenv virtualenv 3.11.10 voice-command-311
```

### Step 3: Set Up Project and Install Python Packages

Now, clone the repository and use the pyenv virtual environment you just created.

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/ruapotato/Voice-Command
cd Voice-Command

# Set the local python version for this directory
pyenv local voice-command-311

# Upgrade pip and install the required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Local LLM Setup

This project uses Ollama for the computer command.

1. Install Ollama from [ollama.com](https://ollama.com).
2. Pull your desired model. For example:

```bash
ollama pull mistral
```

## Running the Application

1. **Ensure Ollama is running**: Before starting the app, make sure the Ollama service is active in the background if you intend to use the computer command.

```bash
ollama serve
```

2. **Navigate and Run**: Open a new terminal and go to the project directory. The pyenv environment should activate automatically. Then, run the main script.

```bash
cd /path/to/Voice-Command
python main.py
```

*Note: The first time you run it, NeMo will download the Parakeet model, which may take some time.*

## Keyboard Controls

* **Record Voice**: Press and hold `Ctrl+Alt`
* **Interrupt/Stop**: Press `Ctrl+C`
* **Exit**: Type `exit` or `quit` at the prompt, or press `Ctrl+D`

## License

GPL3 by David Hamner
