import pyautogui
import pytesseract
from .base import Command

class ClickCommand(Command):
    def __init__(self):
        super().__init__(
            name="click",
            aliases=[],
            description="Click text or buttons on screen",
            execute=self._execute
        )

    async def _execute(self, text: str) -> str:
        """Handle click commands by finding and clicking matching text on screen."""
        try:
            print(f"Searching for text: '{text}'")
            screenshot = pyautogui.screenshot()
            
            # Configure Tesseract for better accuracy
            custom_config = '--psm 11 --oem 3'
            ocr_data = pytesseract.image_to_data(
                screenshot, 
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )
            
            # Debug OCR results
            print("\nOCR Results:")
            found_words = []
            for i, word in enumerate(ocr_data['text']):
                if word.strip():
                    conf = float(ocr_data['conf'][i])
                    found_words.append(f"'{word}' (confidence: {conf:.1f}%)")
            print("Detected words:", ", ".join(found_words[:10]) + "..." if len(found_words) > 10 else ", ".join(found_words))
            
            best_match = None
            highest_confidence = 0
            search_text = text.lower()
            
            for i, word in enumerate(ocr_data['text']):
                if not word.strip():
                    continue
                
                word_lower = word.strip().lower()
                confidence = float(ocr_data['conf'][i])
                
                # Various matching strategies
                matched = False
                match_type = None
                
                if search_text == word_lower:
                    matched = True
                    match_type = "exact"
                    confidence *= 1.2
                elif search_text in word_lower:
                    matched = True
                    match_type = "contains"
                elif word_lower in search_text:
                    matched = True
                    match_type = "partial"
                    confidence *= 0.8
                
                if matched and confidence > highest_confidence:
                    highest_confidence = confidence
                    x = ocr_data['left'][i] + ocr_data['width'][i] // 2
                    y = ocr_data['top'][i] + ocr_data['height'][i] // 2
                    best_match = (x, y, word, match_type, confidence)
            
            if best_match:
                x, y, matched_word, match_type, conf = best_match
                print(f"\nBest match: '{matched_word}' ({match_type} match, confidence: {conf:.1f}%)")
                print(f"Clicking at position: ({x}, {y})")
                
                pyautogui.moveTo(x, y, duration=0.2)
                pyautogui.click()
                
                return f"Clicked '{matched_word}' at ({x}, {y})"
            
            print("\nNo matching text found on screen")
            return "Text not found on screen"
            
        except Exception as e:
            error_msg = f"Click command failed: {str(e)}"
            print(error_msg)
            return error_msg
