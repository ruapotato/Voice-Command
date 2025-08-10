# commands/scrap_command.py
import subprocess
import tempfile
import os
import logging
import shutil
from datetime import datetime
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    pytesseract = None
    Image = None

from .base import Command

logger = logging.getLogger(__name__)

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

class ScrapCommand(Command):
    def __init__(self):
        super().__init__(
            name="scrap",
            aliases=["screengrab", "screen grab", "ocrgrab", "grabtext"],
            description="Select screen area, OCR text, copy to clipboard, and save the image.",
            execute=self._execute
        )
        self.check_dependencies()
        self.pictures_dir = Path.home() / "Pictures" / "scraps"
        os.makedirs(self.pictures_dir, exist_ok=True)

    def check_dependencies(self):
        """Checks for required system tools and libraries."""
        if not PIL_AVAILABLE:
            logger.error("Pillow or Pytesseract not installed. Scrap command disabled.")
            return

        self.screenshot_tool = None
        if is_tool("gnome-screenshot"):
            self.screenshot_tool = "gnome-screenshot"
            logger.info("Using 'gnome-screenshot' for screen capture.")
        elif is_tool("maim"):
             self.screenshot_tool = "maim"
             logger.info("Using 'maim' for screen capture. Ensure 'slop' is installed for selection if needed.")
        elif is_tool("scrot"):
             self.screenshot_tool = "scrot"
             logger.info("Using 'scrot' for screen capture.")
        else:
            logger.error("No suitable screenshot tool found (tried gnome-screenshot, maim, scrot). Scrap command disabled.")

        if not is_tool("xclip"):
            logger.error("'xclip' not found. Scrap command cannot copy to clipboard/primary.")
            self.screenshot_tool = None

    async def _execute(self, args: str) -> str:
        """
        Selects a screen area, performs OCR, copies text, and saves the image.
        Args are ignored.
        """
        if not PIL_AVAILABLE:
            return "Error: Pillow or Pytesseract library not installed."
        if not self.screenshot_tool:
             return "Error: No suitable screenshot tool (gnome-screenshot/maim/scrot) or xclip found."
        if not is_tool("xclip"):
             return "Error: xclip command not found."

        logger.info(f"Starting scrap using {self.screenshot_tool}...")

        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                temp_filename = temp_img_file.name
            logger.debug(f"Temporary screenshot file: {temp_filename}")

            screenshot_success = False
            cmd = []
            if self.screenshot_tool == "gnome-screenshot":
                cmd = ['gnome-screenshot', '-a', '-f', temp_filename]
            elif self.screenshot_tool == "maim":
                 cmd = ['maim', '-s', temp_filename]
            elif self.screenshot_tool == "scrot":
                 cmd = ['scrot', '-s', temp_filename]

            if not cmd:
                 return "Error: Could not determine screenshot command."

            logger.debug(f"Running command: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                    screenshot_success = True
                else:
                     logger.warning(f"{self.screenshot_tool} exited ok, but temp file is missing or empty.")
                     stderr_output = result.stderr.decode('utf-8', errors='ignore').lower()
                     if "cancel" in stderr_output or "giblib error" in stderr_output:
                          screenshot_success = False
                     else:
                          screenshot_success = True
                          logger.warning("Assuming screenshot success despite possible file issue.")

            except FileNotFoundError:
                logger.error(f"Screenshot tool '{self.screenshot_tool}' not found during execution.")
                os.remove(temp_filename)
                return f"Error: Screenshot tool '{self.screenshot_tool}' failed (not found)."
            except subprocess.CalledProcessError as e:
                 logger.info(f"{self.screenshot_tool} exited with error (likely cancelled): {e}")
                 stderr_output = e.stderr.decode('utf-8', errors='ignore').lower()
                 if "cancel" in stderr_output or "giblib error" in stderr_output:
                      screenshot_success = False
                 else:
                      logger.error(f"Screenshot command failed: {e.stderr.decode('utf-8', errors='ignore')}")
                      screenshot_success = False
            except subprocess.TimeoutExpired:
                 logger.error("Screenshot command timed out.")
                 screenshot_success = False
            except Exception as e:
                 logger.error(f"Unexpected error during screenshot: {e}", exc_info=True)
                 screenshot_success = False

            if screenshot_success:
                # Save a copy of the screenshot
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path = self.pictures_dir / f"scrap_{timestamp}.png"
                shutil.copy(temp_filename, save_path)
                logger.info(f"Screenshot saved to {save_path}")

                logger.debug("Screenshot captured successfully. Performing OCR...")
                try:
                    extracted_text = pytesseract.image_to_string(Image.open(temp_filename)).strip()

                    if not extracted_text:
                        logger.info("OCR completed, but no text was found.")
                        os.remove(temp_filename)
                        return "Scrap complete. No text found in the selected area."

                    logger.info(f"OCR successful. Text length: {len(extracted_text)}")
                    logger.debug(f"Extracted text (first 100 chars): {extracted_text[:100]}")

                    try:
                        subprocess.run(
                            ['xclip', '-selection', 'clipboard'],
                            input=extracted_text.encode('utf-8'),
                            check=True,
                            timeout=5
                        )
                        subprocess.run(
                            ['xclip', '-selection', 'primary'],
                             input=extracted_text.encode('utf-8'),
                             check=True,
                             timeout=5
                        )
                        logger.info("Text copied to clipboard and primary selection.")
                        os.remove(temp_filename)
                        return f"Scrap successful. Copied {len(extracted_text)} characters and saved image."

                    except FileNotFoundError:
                         logger.error("xclip not found during copy step.")
                         os.remove(temp_filename)
                         return "Error: xclip not found. Could not copy OCR text."
                    except subprocess.CalledProcessError as e:
                         logger.error(f"xclip command failed: {e.stderr.decode('utf-8', errors='ignore')}")
                         os.remove(temp_filename)
                         return f"Error copying text with xclip: {e.stderr.decode('utf-8', errors='ignore')}"
                    except subprocess.TimeoutExpired:
                         logger.error("xclip command timed out.")
                         os.remove(temp_filename)
                         return "Error: Timeout copying text with xclip."
                    except Exception as e:
                         logger.error(f"Unexpected error during xclip copy: {e}", exc_info=True)
                         os.remove(temp_filename)
                         return f"Error during text copy: {e}"

                except pytesseract.TesseractNotFoundError:
                    logger.error("pytesseract error: 'tesseract' command not found.")
                    os.remove(temp_filename)
                    return "Error: Tesseract OCR engine not found. Please install tesseract-ocr."
                except Exception as ocr_err:
                    logger.error(f"Error during OCR processing: {ocr_err}", exc_info=True)
                    os.remove(temp_filename)
                    return f"Error during OCR: {ocr_err}"
            else:
                 if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                 logger.info("Scrap cancelled or failed before OCR.")
                 return "Scrap cancelled or failed."

        except Exception as outer_e:
            error_msg = f"Unexpected error during scrap: {str(outer_e)}"
            logger.error(error_msg, exc_info=True)
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                 try: os.remove(temp_filename)
                 except Exception as cleanup_e: logger.error(f"Failed to clean up temp file {temp_filename}: {cleanup_e}")
            return error_msg
