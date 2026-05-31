"""
Inference engine for making predictions with a trained model.
Handles text preprocessing, tokenization, image transforms, and model inference.
"""
import torch
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import numpy as np

from src.config import DEVICE, TEXT_MODEL, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, MAX_LEN, MODEL_DIR
from src.models.fusion_model import CyberbullyingDetector
from src.data.preprocessing import TextPreprocessor
from src.utils.scoring import IntentScorer, RepetitionScorer
import os


class InferenceEngine:
    """Run predictions on new text and/or image inputs using a trained model."""

    LABEL_NAMES = ['Aggression', 'Repetition', 'Intent']

    def __init__(self, model_path=None, text_model=None):
        self.device = DEVICE
        text_model = text_model or TEXT_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.preprocessor = TextPreprocessor()
        self.intent_scorer = IntentScorer()

        # Image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])

        # Load model
        self.model = CyberbullyingDetector(text_model_name=text_model)
        model_path = model_path or os.path.join(MODEL_DIR, 'best_model.pth')

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"WARNING: No trained model found at {model_path}. Using untrained model.")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, image_path=None, lang='en'):
        """
        Make a prediction on a single text (and optional image).

        Args:
            text: Input text string
            image_path: Optional path to an image file
            lang: Language ('en', 'roman_urdu', 'urdu')

        Returns:
            Dictionary with predicted labels and scores
        """
        # Preprocess text
        clean_text = self.preprocessor.preprocess(text, lang=lang)
        inputs = self.tokenizer.encode_plus(
            clean_text, None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Preprocess image
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dim

        # Context (default values for single prediction)
        context = torch.tensor([[0.0, 0.0]], dtype=torch.float)

        # Run model
        with torch.no_grad():
            ids = inputs['input_ids'].to(self.device)
            mask = inputs['attention_mask'].to(self.device)
            image_tensor = image_tensor.to(self.device)
            context = context.to(self.device)

            logits = self.model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=None,
                images=image_tensor,
                context=context
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Rule-based intent scoring (supplement model prediction)
        intent_analysis = self.intent_scorer.get_detailed_analysis(text)

        # Build result
        result = {
            'text': text,
            'clean_text': clean_text,
            'language': lang,
            'predictions': {
                self.LABEL_NAMES[i]: {
                    'probability': float(probs[i]),
                    'is_positive': bool(probs[i] >= 0.5)
                }
                for i in range(len(self.LABEL_NAMES))
            },
            'is_cyberbullying': bool(probs[0] >= 0.5),  # Primarily based on aggression
            'intent_analysis': {
                'rule_based_score': intent_analysis['score'],
                'detected_categories': intent_analysis['detected_categories'],
                'matched_keywords': intent_analysis['matched_keywords']
            }
        }

        return result

    def predict_batch(self, texts, lang='en'):
        """Make predictions on a list of texts."""
        results = []
        for text in texts:
            result = self.predict(text, lang=lang)
            results.append(result)
        return results


def format_prediction(result):
    """Pretty-print a prediction result."""
    print(f"\nInput:  \"{result['text']}\"")
    print(f"Clean:  \"{result['clean_text']}\"")
    print(f"Language: {result['language']}")
    print(f"Is Cyberbullying: {'YES' if result['is_cyberbullying'] else 'No'}")
    print(f"\nModel Predictions:")
    for label, data in result['predictions'].items():
        status = "██" if data['is_positive'] else "░░"
        print(f"  {status} {label}: {data['probability']:.3f}")
    print(f"\nIntent Analysis (Rule-based):")
    print(f"  Score: {result['intent_analysis']['rule_based_score']:.3f}")
    if result['intent_analysis']['detected_categories']:
        print(f"  Categories: {', '.join(result['intent_analysis']['detected_categories'])}")
    if result['intent_analysis']['matched_keywords']:
        print(f"  Keywords: {', '.join(result['intent_analysis']['matched_keywords'])}")


if __name__ == "__main__":
    engine = InferenceEngine()

    # Test with different languages
    examples = [
        ("You are a waste of space, nobody likes you!", 'en'),
        ("Tum bohut buray insaan ho, khabardar!", 'roman_urdu'),
        ("Let's meet at the park tomorrow", 'en'),
        ("Tujhe dekh lunga, mar dunga", 'roman_urdu'),
    ]

    for text, lang in examples:
        result = engine.predict(text, lang=lang)
        format_prediction(result)
        print("-" * 60)
