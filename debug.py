"""Debug script to understand data structure"""
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('xlsx_path', help='Path to the split .xlsx file')
parser.add_argument('--sheet', default='Men', help='Sheet name to inspect (default: Men)')
args = parser.parse_args()

df = pd.read_excel(args.xlsx_path, sheet_name=args.sheet)
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
