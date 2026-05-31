import pandas as pd

try:
    df = pd.read_excel('data/Roman annotated data.xlsx')
    print(df.columns.tolist())
except Exception as e:
    print(f"Error: {e}")
