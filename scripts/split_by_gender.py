"""
Script to split Excel data into separate sheets by gender.

Usage:
    python split_by_gender.py --input "./files/data/26.1/data.xlsx" --women-file "./women_names.txt"

The women_names file should have one name per line:
    Maria Santos
    Alaya Minet
    Ana Paula

Everyone NOT in the list will be placed in the "Men" sheet.
"""
import pandas as pd
import argparse
import os


def load_women_names(names_file: str) -> set:
    """
    Load women's names from a file (one name per line).
    
    Returns:
        Set of normalized women's names
    """
    names = set()
    
    with open(names_file, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name.lower())
    
    return names


def is_woman(name: str, women_names: set) -> bool:
    """Check if a name is in the women's list."""
    if pd.isna(name):
        return False
    return str(name).strip().lower() in women_names


def split_by_gender(
    input_file: str,
    output_file: str,
    women_names: set,
    name_column: str = "Tell us what's you name: ",
    sheet_name: str = None
) -> dict:
    """
    Split Excel file into separate sheets by gender.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output Excel file (will have 'Men' and 'Women' sheets)
        women_names: Set of women's names (normalized to lowercase)
        name_column: Column containing participant names
        sheet_name: Optional specific sheet to read (None = first sheet)
        
    Returns:
        Dict with statistics about the split
    """
    # Read the Excel file
    if sheet_name:
        df = pd.read_excel(input_file, sheet_name=sheet_name)
    else:
        df = pd.read_excel(input_file)
    
    print(f"Loaded {len(df)} rows from {input_file}")
    
    # Find the name column
    if name_column not in df.columns:
        possible_cols = [c for c in df.columns if 'name' in c.lower()]
        if possible_cols:
            name_column = possible_cols[0]
            print(f"Using column '{name_column}' for names")
        else:
            raise ValueError(f"Name column '{name_column}' not found. Available: {list(df.columns)}")
    
    # Split based on women's names
    is_woman_mask = df[name_column].apply(lambda x: is_woman(x, women_names))
    
    df_women = df[is_woman_mask]
    df_men = df[~is_woman_mask]
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_men.to_excel(writer, sheet_name='Men', index=False)
        df_women.to_excel(writer, sheet_name='Women', index=False)
    
    stats = {
        'total': len(df),
        'men': len(df_men),
        'women': len(df_women),
    }
    
    print(f"\nResults:")
    print(f"  Total: {stats['total']}")
    print(f"  Men: {stats['men']}")
    print(f"  Women: {stats['women']}")
    print(f"\nOutput saved to: {output_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Split Excel file into Men/Women sheets"
    )
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True,
        help="Path to input Excel file"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default=None,
        help="Path to output Excel file (default: input_split.xlsx)"
    )
    parser.add_argument(
        "--women-file", "-w", 
        type=str, 
        required=True,
        help="Path to file with women's names (one per line)"
    )
    parser.add_argument(
        "--name-column", 
        type=str, 
        default="Tell us what's you name: ",
        help="Column name containing participant names"
    )
    parser.add_argument(
        "--sheet", 
        type=str, 
        default=None,
        help="Specific sheet to read from input file"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_split{ext}"
    
    # Load women's names
    print(f"Loading women's names from {args.women_file}...")
    women_names = load_women_names(args.women_file)
    print(f"Loaded {len(women_names)} women's names")
    
    # Split the file
    split_by_gender(
        input_file=args.input,
        output_file=args.output,
        women_names=women_names,
        name_column=args.name_column,
        sheet_name=args.sheet
    )


if __name__ == "__main__":
    main()

