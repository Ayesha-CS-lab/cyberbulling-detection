import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, AutoModel

class CyberbullyingDetector(nn.Module):
    def __init__(self, text_model_name='bert-base-multilingual-cased', num_classes=3):
        super(CyberbullyingDetector, self).__init__()
        
        # 1. Text Encoder (m-BERT / MuRIL)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size # Usually 768
        
        # 2. Image Encoder (ResNet50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1] # Remove classification layer
        self.image_encoder = nn.Sequential(*modules)
        image_hidden_size = 2048
        
        # 3. Context Encoder (Simple MLP)
        self.context_encoder = nn.Sequential(
            nn.Linear(2, 64), # 2 features: age, past_flags
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        context_hidden_size = 64
        
        # 4. Fusion Layer (Attention Mechanism)
        self.fusion_dim = 512
        
        # Project all modalities to same dimension
        self.text_proj = nn.Linear(text_hidden_size, self.fusion_dim)
        self.image_proj = nn.Linear(image_hidden_size, self.fusion_dim)
        self.context_proj = nn.Linear(context_hidden_size, self.fusion_dim)
        
        # Attention weights
        self.attention_weights = nn.Linear(self.fusion_dim, 1)
        
        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes) # [Aggression, Repetition, Intent]
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids, images, context):
        # Text Features
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        text_feat = text_out.pooler_output # [Batch, 768]
        
        # Image Features
        image_feat = self.image_encoder(images) # [Batch, 2048, 1, 1]
        image_feat = image_feat.view(image_feat.size(0), -1) # Flatten
        
        # Context Features
        context_feat = self.context_encoder(context)
        
        # Projection
        text_emb = self.text_proj(text_feat)
        image_emb = self.image_proj(image_feat)
        context_emb = self.context_proj(context_feat)
        
        # Stack Features: [Batch, 3, FusionDim]
        features = torch.stack([text_emb, image_emb, context_emb], dim=1)
        
        # Attention (Simple Dot Product)
        # scores = [Batch, 3, 1]
        attn_scores = torch.tanh(self.attention_weights(features))
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Weighted Sum
        fused_feat = torch.sum(features * attn_weights, dim=1) # [Batch, FusionDim]
        
        # Classification
        logits = self.classifier(fused_feat)
        
        return logits

# Note: MuRIL model name: 'google/muril-base-cased'
