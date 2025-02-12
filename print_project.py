#!/usr/bin/env python3
"""Print all project files in a nicely formatted way"""

from pathlib import Path
import sys
from typing import List, Set

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

def print_header(title: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(HEADER_TEMPLATE.format(
        title=title,
        underline=char * len(title)
    ))

def print_file_content(file_path: Path) -> None:
    """Print the content of a file with nice formatting."""
    try:
        content = file_path.read_text()
        print(FILE_TEMPLATE.format(
            filename=file_path,
            separator="-" * 80,
            content=content
        ))
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)

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

def print_directory_structure(directory: Path, prefix: str = "") -> None:
    """Print the directory structure in a tree-like format."""
    try:
        items = sorted(directory.iterdir())
        for i, item in enumerate(items):
            if not should_process(item):
                continue
                
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            
            print(f"{prefix}{current_prefix}{item.name}")
            if item.is_dir():
                print_directory_structure(item, prefix + next_prefix)
    except Exception as e:
        print(f"Error accessing {directory}: {e}", file=sys.stderr)

def main():
    """Main function to print project files."""
    project_root = Path(__file__).parent
    
    print_header("Project Structure")
    print(f"Root: {project_root}")
    print_directory_structure(project_root)
    print()
    
    print_header("Project Files")
    for file_path in find_all_files(project_root):
        if file_path.suffix in ['.py', '.txt', '.md', '.json', '.yaml', '.yml']:
            print_file_content(file_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrinting interrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
