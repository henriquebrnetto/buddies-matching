"""Quick script to split the data by gender."""
import argparse
import glob
import os
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder', help='Folder containing the input .xlsx and women_names.txt')
parser.add_argument('-o', '--output-folder', default=None, help='Folder to save data_split.xlsx (defaults to the input folder)')
args = parser.parse_args()
output_folder = args.output_folder or args.folder

# Find the Excel file
input_file = glob.glob(os.path.join(args.folder, '*.xlsx'))[0]
print(f"Reading: {input_file}")

# Load data
df = pd.read_excel(input_file)
print(f"Total rows: {len(df)}")

# Load women's names
women_names_file = os.path.join(args.folder, 'women_names.txt')
with open(women_names_file, 'r', encoding='utf-8') as f:
    women_names = set(line.strip().lower() for line in f if line.strip())
print(f"Women names loaded: {len(women_names)}")

# Find name column
name_col = "Tell us what's you name: "
if name_col not in df.columns:
    name_col = [c for c in df.columns if 'name' in c.lower()][0]
print(f"Using name column: {name_col}")

# Split
def is_woman(name):
    if pd.isna(name):
        return False
    return str(name).strip().lower() in women_names

mask = df[name_col].apply(is_woman)
df_women = df[mask]
df_men = df[~mask]

print(f"\nResults:")
print(f"  Women: {len(df_women)}")
print(f"  Men: {len(df_men)}")

# Save
output_file = os.path.join(output_folder, 'data_split.xlsx')
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_men.to_excel(writer, sheet_name='Men', index=False)
    df_women.to_excel(writer, sheet_name='Women', index=False)

print(f"\nSaved to: {output_file}")
