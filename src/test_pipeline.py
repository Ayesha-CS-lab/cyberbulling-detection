import torch
from src.models.fusion_model import CyberbullyingDetector

def test_pipeline():
    print("Testing Model Pipeline...")
    
    # 1. Initialize Model
    try:
        model = CyberbullyingDetector()
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return

    # 2. Create Dummy Inputs
    batch_size = 2
    # BERT Inputs
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones((batch_size, 128))
    token_type_ids = torch.zeros((batch_size, 128), dtype=torch.long)
    
    # Image Inputs (normalized [0,1])
    images = torch.rand((batch_size, 3, 224, 224))
    
    # Context Inputs
    context = torch.rand((batch_size, 2))
    
    # 3. Forward Pass
    try:
        print("Running forward pass...")
        outputs = model(input_ids, attention_mask, token_type_ids, images, context)
        print(f"Forward pass successful. Output shape: {outputs.shape}")
        
        # Check output dimension: [Batch, 3] (Aggression, Repetition, Intent)
        assert outputs.shape == (batch_size, 3)
        print("Output shape is correct.")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_pipeline()
