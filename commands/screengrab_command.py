import subprocess
import tempfile
import os
import logging
import shutil # For checking command existence

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

# --- Helper function to check for command existence ---
def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

# --- ScreengrabCommand Class ---
class ScreengrabCommand(Command):
    def __init__(self):
        super().__init__(
            name="screengrab",
            aliases=["ocrgrab", "grabtext"],
            description="Select screen area, OCR text, copy to clipboard & primary selection.",
            execute=self._execute
        )
        self.check_dependencies()

    def check_dependencies(self):
        """Checks for required system tools and libraries."""
        if not PIL_AVAILABLE:
            logger.error("Pillow or Pytesseract not installed. Screengrab command disabled.")
            return

        self.screenshot_tool = None
        # Prioritize gnome-screenshot (interactive area selection)
        if is_tool("gnome-screenshot"):
            self.screenshot_tool = "gnome-screenshot"
            logger.info("Using 'gnome-screenshot' for screen capture.")
        # Fallback using maim (needs slop typically, but -s might work alone)
        elif is_tool("maim"):
             self.screenshot_tool = "maim"
             logger.info("Using 'maim' for screen capture. Ensure 'slop' is installed for selection if needed.")
        # Fallback using scrot (older tool)
        elif is_tool("scrot"):
             self.screenshot_tool = "scrot"
             logger.info("Using 'scrot' for screen capture.")
        else:
            logger.error("No suitable screenshot tool found (tried gnome-screenshot, maim, scrot). Screengrab command disabled.")

        if not is_tool("xclip"):
            logger.error("'xclip' not found. Screengrab command cannot copy to clipboard/primary.")
            # Allow command to potentially still run OCR, but warn user? Or disable fully?
            # Let's disable for now as copying is key part of description.
            self.screenshot_tool = None # Mark as unusable

    async def _execute(self, args: str) -> str:
        """
        Selects a screen area, performs OCR, and copies text.
        Args are ignored.
        """
        if not PIL_AVAILABLE:
            return "Error: Pillow or Pytesseract library not installed."
        if not self.screenshot_tool:
             return "Error: No suitable screenshot tool (gnome-screenshot/maim/scrot) or xclip found."
        if not is_tool("xclip"): # Double check xclip
             return "Error: xclip command not found."

        logger.info(f"Starting screengrab using {self.screenshot_tool}...")

        try:
            # Create a temporary file for the screenshot
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                temp_filename = temp_img_file.name
            logger.debug(f"Temporary screenshot file: {temp_filename}")

            # --- Run Screenshot Tool ---
            screenshot_success = False
            cmd = []
            if self.screenshot_tool == "gnome-screenshot":
                # -a for area, -f for file
                cmd = ['gnome-screenshot', '-a', '-f', temp_filename]
            elif self.screenshot_tool == "maim":
                 # -s for select, outputs PNG to stdout by default, redirect or specify file
                 # Using -s requires slop usually, let's try direct file output with selection
                 cmd = ['maim', '-s', temp_filename]
                 # Note: If maim -s hangs without slop, this needs adjustment
                 # A common pattern is: maim -o -s | xclip -selection clipboard -t image/png
                 # but we need the file for OCR. maim $(slop) file.png is better if slop exists.
            elif self.screenshot_tool == "scrot":
                 # -s for select, specify file at the end
                 cmd = ['scrot', '-s', temp_filename]

            if not cmd: # Should not happen if checks pass, but safeguard
                 return "Error: Could not determine screenshot command."

            logger.debug(f"Running command: {' '.join(cmd)}")
            try:
                # Execute the screenshot command
                result = subprocess.run(cmd, check=True, capture_output=True, timeout=60) # Generous timeout for selection
                # Check if the file was actually created and has size
                if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                    screenshot_success = True
                else:
                     logger.warning(f"{self.screenshot_tool} exited ok, but temp file is missing or empty.")
                     # scrot might put filename in stdout/stderr on cancel? Check result.
                     stderr_output = result.stderr.decode('utf-8', errors='ignore').lower()
                     if "cancel" in stderr_output or "giblib error" in stderr_output: # Check common cancel messages
                          screenshot_success = False
                     else: # Assume success but warn
                          screenshot_success = True
                          logger.warning("Assuming screenshot success despite possible file issue.")


            except FileNotFoundError:
                logger.error(f"Screenshot tool '{self.screenshot_tool}' not found during execution.")
                os.remove(temp_filename) # Clean up empty temp file
                return f"Error: Screenshot tool '{self.screenshot_tool}' failed (not found)."
            except subprocess.CalledProcessError as e:
                 # This often indicates cancellation by the user (e.g., pressing Esc)
                 logger.info(f"{self.screenshot_tool} exited with error (likely cancelled): {e}")
                 stderr_output = e.stderr.decode('utf-8', errors='ignore').lower()
                 if "cancel" in stderr_output or "giblib error" in stderr_output: # Check common cancel messages
                      screenshot_success = False
                 else:
                      # Treat other errors as failures
                      logger.error(f"Screenshot command failed: {e.stderr.decode('utf-8', errors='ignore')}")
                      screenshot_success = False
            except subprocess.TimeoutExpired:
                 logger.error("Screenshot command timed out.")
                 screenshot_success = False
            except Exception as e: # Catch unexpected errors
                 logger.error(f"Unexpected error during screenshot: {e}", exc_info=True)
                 screenshot_success = False


            # --- Process Screenshot if Successful ---
            if screenshot_success:
                logger.debug("Screenshot captured successfully. Performing OCR...")
                try:
                    # Perform OCR using Pytesseract
                    extracted_text = pytesseract.image_to_string(Image.open(temp_filename)).strip()

                    if not extracted_text:
                        logger.info("OCR completed, but no text was found.")
                        os.remove(temp_filename) # Clean up image
                        return "Screengrab complete. No text found in the selected area."

                    logger.info(f"OCR successful. Text length: {len(extracted_text)}")
                    logger.debug(f"Extracted text (first 100 chars): {extracted_text[:100]}")

                    # --- Copy to Clipboard and Primary Selection ---
                    try:
                        # Copy to standard clipboard
                        subprocess.run(
                            ['xclip', '-selection', 'clipboard'],
                            input=extracted_text.encode('utf-8'),
                            check=True,
                            timeout=5
                        )
                        # Copy to primary selection (highlight buffer)
                        subprocess.run(
                            ['xclip', '-selection', 'primary'],
                             input=extracted_text.encode('utf-8'),
                             check=True,
                             timeout=5
                        )
                        logger.info("Text copied to clipboard and primary selection.")
                        os.remove(temp_filename) # Clean up image
                        return f"Screengrab successful. Copied {len(extracted_text)} characters."

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
                 # Screenshot failed or was cancelled
                 if os.path.exists(temp_filename): # Clean up if file exists but wasn't processed
                    os.remove(temp_filename)
                 logger.info("Screengrab cancelled or failed before OCR.")
                 return "Screengrab cancelled or failed."

        except Exception as outer_e:
            # Catch any other unexpected errors in the process
            error_msg = f"Unexpected error during screengrab: {str(outer_e)}"
            logger.error(error_msg, exc_info=True)
            # Ensure cleanup if temp_filename was assigned
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                 try: os.remove(temp_filename)
                 except Exception as cleanup_e: logger.error(f"Failed to clean up temp file {temp_filename}: {cleanup_e}")
            return error_msg
