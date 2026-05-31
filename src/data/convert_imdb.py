import pandas as pd
import numpy as np
import os

def convert_imdb_dataset(input_file, output_file):
    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("Mapping sentiment to aggression labels...")
    # Map negative -> 1 (Aggressive/Toxic proxy), positive -> 0
    df['aggression'] = df['sentiment'].apply(lambda x: 1 if str(x).lower() == 'negative' else 0)
    
    print(df['aggression'].value_counts())

    # Construct New DataFrame
    new_df = pd.DataFrame()
    new_df['text_content'] = df['review']
    
    # Missing Modalities
    new_df['image_filename'] = 'none.jpg'
    new_df['user_age'] = 0
    new_df['past_flags'] = 0
    
    # Targets
    new_df['aggression'] = df['aggression']
    new_df['repetition'] = 0 
    new_df['intent'] = df['aggression'] # Assumption: formatting negative sentiment as harmful intent for pre-training
    
    # Metadata
    new_df['language'] = 'en'

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_df.to_csv(output_file, index=False)
    print(f"Saved IMDb converted dataset to {output_file} with {len(new_df)} rows.")

if __name__ == "__main__":
    convert_imdb_dataset('data/IMDB Dataset.csv', 'data/train_imdb.csv')
