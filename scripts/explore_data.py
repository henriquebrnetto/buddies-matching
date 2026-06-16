import argparse
import pandas as pd
import glob
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

parser = argparse.ArgumentParser(description='Explore the structure of an input data file')
parser.add_argument('data_dir', help='Folder containing the input .xlsx file')
parser.add_argument('--pattern', default='*2026*.xlsx', help='Glob pattern to find the file (default: *2026*.xlsx)')
parser.add_argument('-o', '--output', default='data_structure.txt', help='Output file (default: data_structure.txt)')
args = parser.parse_args()

# Find the Excel file
files = glob.glob(os.path.join(args.data_dir, args.pattern))
file_path = files[0] if files else None
df = pd.read_excel(file_path)

with open(args.output, 'w', encoding='utf-8') as f:
    f.write('Columns:\n')
    for i, col in enumerate(df.columns):
        f.write(f"  {i}: {col}\n")
    f.write(f'\nShape: {df.shape}\n')
    f.write('\nFirst row sample:\n')
    for col in df.columns:
        f.write(f"  {col}: {repr(df[col].iloc[0])}\n")
    f.write('\n\nUnique values per column:\n')
    for col in df.columns:
        unique_vals = df[col].dropna().unique()[:5]  # First 5 unique values
        f.write(f"  {col}: {list(unique_vals)}\n")
    f.write('\n\nComments column (if exists):\n')
    if 'Comments' in df.columns:
        for i, comment in enumerate(df['Comments'].dropna().values[:10]):
            f.write(f"  {i}: {comment}\n")
print(f'Output written to {args.output}')
