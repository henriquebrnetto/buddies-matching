"""Quick script to split the data by gender."""
import glob
import pandas as pd

# Find the Excel file
input_file = glob.glob('files/data/26.1/*.xlsx')[0]
print(f"Reading: {input_file}")

# Load data
df = pd.read_excel(input_file)
print(f"Total rows: {len(df)}")

# Load women's names
with open('files/data/26.1/women_names.txt', 'r', encoding='utf-8') as f:
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
output_file = 'files/data/26.1/data_split.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_men.to_excel(writer, sheet_name='Men', index=False)
    df_women.to_excel(writer, sheet_name='Women', index=False)

print(f"\nSaved to: {output_file}")
