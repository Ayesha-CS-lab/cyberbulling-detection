"""
Stage 1 model: a text-only multilingual aggression classifier.

This is the message-level building block of the methodology (Figure 1):
m-BERT / MuRIL encodes a single message and predicts whether it is aggressive.
Repetition, intent and the final cyberbullying decision are handled at the
user-pair level in Stage 2 (see src/train_stage2.py).
"""
import torch.nn as nn
from transformers import AutoModel


class AggressionClassifier(nn.Module):
    def __init__(self, text_model_name='bert-base-multilingual-cased', dropout=0.3):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        hidden = self.text_encoder.config.hidden_size  # 768 for m-BERT / MuRIL
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)  # single logit -> aggressive / not

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = out.pooler_output            # [B, hidden]
        logit = self.classifier(self.dropout(pooled))  # [B, 1]
        return logit.squeeze(-1)              # [B]
