"""Debug script to understand data structure"""
import pandas as pd

df = pd.read_excel('files/data/26.1/data_split.xlsx', sheet_name='Men')
print("=== Column structure ===")
for i, col in enumerate(df.columns):
    print(f"{i}: {repr(col)}")

print(f"\n=== Looking for key columns ===")
# Find name column
name_cols = [c for c in df.columns if 'name' in c.lower()]
print(f"Name columns: {name_cols}")

# Find type column
type_cols = [c for c in df.columns if 'am a' in c.lower()]
print(f"Type columns: {type_cols}")

if type_cols:
    print(f"\n=== Type column values ===")
    print(df[type_cols[0]].value_counts())
