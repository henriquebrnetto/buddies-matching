"""Quick analysis of buddy/international split."""
import pandas as pd

df = pd.read_excel('files/data/26.1/data_split.xlsx', sheet_name=None)

# Find the "I am a" column (may have special chars)
def get_type_col(sheet):
    return [c for c in sheet.columns if 'I am a' in c][0]

print("=== WOMEN ===")
w = df['Women']
type_col = get_type_col(w)
print(w[type_col].value_counts())

print("\n=== MEN ===")
m = df['Men']
type_col_m = get_type_col(m)
print(m[type_col_m].value_counts())

print("\n=== SUMMARY ===")
w_buddy = len(w[w[type_col].str.contains('Buddy', na=False)])
w_int = len(w[w[type_col].str.contains('International', na=False)])
m_buddy = len(m[m[type_col_m].str.contains('Buddy', na=False)])
m_int = len(m[m[type_col_m].str.contains('International', na=False)])

print(f"Women Buddies: {w_buddy}")
print(f"Women International: {w_int}")
print(f"Women Ratio: {w_int/w_buddy:.2f} students/buddy")
print()
print(f"Men Buddies: {m_buddy}")
print(f"Men International: {m_int}")
print(f"Men Ratio: {m_int/m_buddy:.2f} students/buddy")

