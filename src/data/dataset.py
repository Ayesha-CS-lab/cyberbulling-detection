import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from .preprocessing import TextPreprocessor


def get_image_transforms(image_size=224):
    """Standard ImageNet-normalized transforms for the image encoder."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class CyberbullyingDataset(Dataset):
    def __init__(self, metadata_file, image_dir, tokenizer, transform=None, max_len=128):
        """
        Args:
            metadata_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            tokenizer (callable): HuggingFace tokenizer.
            transform (callable, optional): Optional transform for images.
                If None, uses default ImageNet transforms.
            max_len (int): Maximum length for BERT tokenization.
        """
        self.data = pd.read_csv(metadata_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform if transform else get_image_transforms()
        self.max_len = max_len
        self.preprocessor = TextPreprocessor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]

        # 1. Process Text
        text = str(row['text_content'])
        lang = row.get('language', 'en')
        clean_text = self.preprocessor.preprocess(text, lang=lang)

        inputs = self.tokenizer.encode_plus(
            clean_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs.get('token_type_ids', [0] * self.max_len)

        # 2. Process Image
        img_name = os.path.join(self.image_dir, str(row['image_filename']))
        image = Image.new('RGB', (224, 224))  # Default blank image
        try:
            if os.path.exists(img_name):
                image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")

        # Apply transforms (resize, normalize, to tensor)
        image = self.transform(image)

        # 3. Process Context
        context_features = torch.tensor([
            float(row.get('user_age', 0)),
            float(row.get('past_flags', 0))
        ], dtype=torch.float)

        # 4. Process Labels: [Aggression, Repetition, Intent]
        labels = torch.tensor([
            int(row.get('aggression', 0)),
            int(row.get('repetition', 0)),
            int(row.get('intent', 0))
        ], dtype=torch.float)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'image': image,
            'context': context_features,
            'targets': labels
        }
