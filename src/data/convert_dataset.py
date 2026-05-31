import pandas as pd
import numpy as np
import os

def convert_roman_dataset(input_file, output_file):
    print(f"Reading {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 1. Consensus Voting for Aggression
    # Annotators might have 1, 0, or NaN. Treat NaN as 0.
    a1 = df['Annotator 1'].fillna(0).astype(int)
    a2 = df['Annotator 2'].fillna(0).astype(int)
    a3 = df['Annotator 3'].fillna(0).astype(int)
    
    # Sum votes
    votes = a1 + a2 + a3
    # If 2 or more votes, class is 1 (Aggressive)
    aggression_labels = (votes >= 2).astype(int)
    
    print(f"Labels distribution:\n{aggression_labels.value_counts()}")

    # 2. Construct New DataFrame
    new_df = pd.DataFrame()
    new_df['text_content'] = df['Text'].astype(str)
    
    # Missing Modalities
    new_df['image_filename'] = 'none.jpg' # Placeholder
    new_df['user_age'] = 0
    new_df['past_flags'] = 0
    
    # Targets
    new_df['aggression'] = aggression_labels
    new_df['repetition'] = 0 # Not present in dataset
    new_df['intent'] = 0     # Not present in dataset
    
    # Metadata
    new_df['language'] = 'roman_urdu' # Assumption based on filename

    # 3. Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_df.to_csv(output_file, index=False)
    print(f"Saved converted dataset to {output_file} with {len(new_df)} rows.")

if __name__ == "__main__":
    convert_roman_dataset('data/Roman annotated data.xlsx', 'data/train.csv')
