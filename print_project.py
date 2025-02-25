#!/usr/bin/env python3
"""Print all project files in a nicely formatted way and copy to clipboard"""
from pathlib import Path
import sys
from typing import List, Set
import io
import subprocess
from shutil import which

HEADER_TEMPLATE = """
{title}
{underline}
"""

FILE_TEMPLATE = """
File: {filename}
{separator}
{content}
"""

# Files to ignore
IGNORE_PATTERNS = {
    '__pycache__',
    '.git',
    '.pyc',
    '.env',
    'pyenv',
    '.vscode',
    '.idea'
}

def should_process(path: Path) -> bool:
    """Check if the path should be processed."""
    return not any(ignore in str(path) for ignore in IGNORE_PATTERNS)

def print_header(title: str, char: str = "=") -> str:
    """Print a formatted header and return it as a string."""
    header = HEADER_TEMPLATE.format(
        title=title,
        underline=char * len(title)
    )
    print(header)
    return header

def print_file_content(file_path: Path) -> str:
    """Print the content of a file with nice formatting and return it as a string."""
    try:
        content = file_path.read_text()
        formatted = FILE_TEMPLATE.format(
            filename=file_path,
            separator="-" * 80,
            content=content
        )
        print(formatted)
        return formatted
    except Exception as e:
        error_msg = f"Error reading {file_path}: {e}"
        print(error_msg, file=sys.stderr)
        return error_msg

def find_all_files(directory: Path) -> List[Path]:
    """Recursively find all files in the directory."""
    files = []
    try:
        for item in directory.iterdir():
            if not should_process(item):
                continue
                
            if item.is_file():
                files.append(item)
            elif item.is_dir():
                files.extend(find_all_files(item))
    except Exception as e:
        print(f"Error accessing {directory}: {e}", file=sys.stderr)
    
    return sorted(files)

def print_directory_structure(directory: Path, prefix: str = "") -> str:
    """Print the directory structure in a tree-like format and return it as a string."""
    output = []
    try:
        items = sorted(directory.iterdir())
        for i, item in enumerate(items):
            if not should_process(item):
                continue
                
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            
            line = f"{prefix}{current_prefix}{item.name}"
            print(line)
            output.append(line)
            
            if item.is_dir():
                dir_output = print_directory_structure(item, prefix + next_prefix)
                output.append(dir_output)
    except Exception as e:
        error_msg = f"Error accessing {directory}: {e}"
        print(error_msg, file=sys.stderr)
        output.append(error_msg)
    
    return "\n".join(output)

def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard based on the platform."""
    try:
        # macOS
        if which('pbcopy'):
            subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)
            return True
        # Linux with xclip
        elif which('xclip'):
            subprocess.run(['xclip', '-selection', 'clipboard'], input=text.encode('utf-8'), check=True)
            return True
        # Linux with wl-copy (Wayland)
        elif which('wl-copy'):
            subprocess.run(['wl-copy'], input=text.encode('utf-8'), check=True)
            return True
        # Windows
        elif sys.platform == 'win32':
            import pyperclip
            pyperclip.copy(text)
            return True
        else:
            print("Could not find a suitable clipboard tool. Consider installing pyperclip.", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Failed to copy to clipboard: {e}", file=sys.stderr)
        return False

def main():
    """Main function to print project files and copy to clipboard."""
    # Use StringIO to capture all output
    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    project_root = Path(__file__).parent
    
    print_header("Project Structure")
    print(f"Root: {project_root}")
    print_directory_structure(project_root)
    print()
    
    print_header("Project Files")
    for file_path in find_all_files(project_root):
        if file_path.suffix in ['.py', '.txt', '.md', '.json', '.yaml', '.yml']:
            print_file_content(file_path)
    
    # Restore stdout and get the captured output
    sys.stdout = original_stdout
    full_output = output_buffer.getvalue()
    
    # Print the output to the terminal
    print(full_output)
    
    # Copy to clipboard
    if copy_to_clipboard(full_output):
        print("\nProject structure and files have been copied to clipboard!")
    else:
        print("\nFailed to copy to clipboard. You may need to install a clipboard package.")
        print("For Python, you can use: pip install pyperclip")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrinting interrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
