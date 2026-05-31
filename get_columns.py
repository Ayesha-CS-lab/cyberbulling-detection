import pandas as pd

try:
    df = pd.read_excel('data/Roman annotated data.xlsx')
    with open('columns.txt', 'w', encoding='utf-8') as f:
        for col in df.columns:
            f.write(f"{col}\n")
    print("Columns written to columns.txt")
except Exception as e:
    print(f"Error: {e}")
