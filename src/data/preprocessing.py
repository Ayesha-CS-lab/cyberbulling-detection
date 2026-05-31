import re
import unicodedata

class TextPreprocessor:
    def __init__(self):
        # Basic cleaning patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.user_mention_pattern = re.compile(r'@\w+')
        self.emoji_pattern = re.compile(r'[^\w\s,.]') # Crude emoji removal, can be refined

    def clean_text(self, text):
        """
        Cleans text by removing URLs, HTML tags, user mentions, and special characters.
        """
        if not isinstance(text, str):
            return ""
        
        text = self.url_pattern.sub('', text)
        text = self.html_pattern.sub('', text)
        text = self.user_mention_pattern.sub('', text)
        text = self.emoji_pattern.sub(' ', text) # Replace unknown chars with space
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        return text

    def normalize_roman_urdu(self, text):
        """
        Normalizes Roman Urdu text. 
        This is a placeholder for a more complex transliteration/normalization logic.
        For now, it lowercases and removes some common noise.
        """
        text = text.lower()
        # Example: ambiguous spellings normalization (very basic)
        text = text.replace("aa", "a").replace("ee", "i").replace("oo", "u")
        return text

    def preprocess(self, text, lang='en'):
        """
        Main preprocessing pipeline.
        """
        cleaned = self.clean_text(text)
        if lang == 'roman_urdu':
            cleaned = self.normalize_roman_urdu(cleaned)
        return cleaned

if __name__ == "__main__":
    # Test
    processor = TextPreprocessor()
    sample = "Hello @user! check this link https://test.com/ image"
    print(f"Original: {sample}")
    print(f"Cleaned: {processor.preprocess(sample)}")
    
    roman_sample = "tum bohut buray ho"
    print(f"Original: {roman_sample}")
    print(f"Normalized: {processor.preprocess(roman_sample, lang='roman_urdu')}")
