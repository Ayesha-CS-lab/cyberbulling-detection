"""
CLIP-based multimodal classifier for image + text (meme) cyberbullying detection.

Rationale (thesis §5.4): the earlier ResNet50 + m-BERT model overfit and collapsed
because two large backbones were fully fine-tuned on a few thousand memes. Here the
CLIP backbone is FROZEN and only a small classification head is trained on top of
CLIP's jointly-learned image and text embeddings — the standard strong recipe for
hateful/harmful-meme detection on limited data.

`mode` selects the ablation:
    'fusion' -> concat(image_emb, text_emb)   (the multimodal model)
    'image'  -> image_emb only                (image-only baseline)
    'text'   -> text_emb only                 (text-only baseline)
"""
import torch
import torch.nn as nn


class CLIPClassifier(nn.Module):
    def __init__(self, clip_name='openai/clip-vit-base-patch32', mode='fusion', dropout=0.3):
        super().__init__()
        from transformers import CLIPModel  # lazy: only needed when the full model is used
        self.clip = CLIPModel.from_pretrained(clip_name)
        for p in self.clip.parameters():          # freeze the backbone
            p.requires_grad = False
        self.clip.eval()
        d = self.clip.config.projection_dim        # 512 for ViT-B/32
        self.mode = mode
        in_dim = {'fusion': 2 * d, 'image': d, 'text': d}[mode]
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _pool(out):
        p = getattr(out, 'pooler_output', None)
        return p if p is not None else out.last_hidden_state[:, 0]

    @torch.no_grad()
    def encode(self, pixel_values, input_ids, attention_mask):
        vout = self.clip.vision_model(pixel_values=pixel_values)
        tout = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        img = self.clip.visual_projection(self._pool(vout))
        txt = self.clip.text_projection(self._pool(tout))
        img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return img, txt

    def head_forward(self, img, txt):
        if self.mode == 'fusion':
            feat = torch.cat([img, txt], dim=-1)
        elif self.mode == 'image':
            feat = img
        else:
            feat = txt
        return self.head(feat).squeeze(-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        img, txt = self.encode(pixel_values, input_ids, attention_mask)
        return self.head_forward(img, txt)


class HeadOnly(nn.Module):
    """Small classifier trained on PRE-COMPUTED CLIP features (fast iteration)."""
    def __init__(self, in_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
