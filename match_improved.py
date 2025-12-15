"""
Improved buddy matching entry point with configurable options.
"""
from classes.improved_optimizer import ImprovedOptimizer
from classes.optimizer import Optimizer
from models.config import MatchingConfig
from utils import clean_dataframe, split_dfs, get_info_dict, preprocessing
import pandas as pd
from argparse import ArgumentParser
import os


def find_column(df: pd.DataFrame, search_text: str) -> str | None:
    """Find a column containing the search text (handles special chars like \xa0)."""
    for col in df.columns:
        if search_text.lower() in col.lower().replace('\xa0', ' '):
            return col
    return None


def main(
    to_csv: bool = False, 
    to_excel: bool = False, 
    save_path: str = ".", 
    sexo: str = "m", 
    classes: list[str] = ['buddy', 'int'], 
    file_path: str = "data/dados.xlsx", 
    sheet: str = "Sheet1",
    use_improved: bool = True,
    config: MatchingConfig = None
) -> ImprovedOptimizer | Optimizer:
    """
    Main function for buddy matching.
    
    Args:
        to_csv: Save results to CSV
        to_excel: Save results to Excel  
        save_path: Directory to save results
        sexo: Gender identifier ('m' or 'h')
        classes: List of class types ['buddy', 'int']
        file_path: Path to Excel file with data
        sheet: Sheet name in Excel file
        use_improved: Use improved optimizer with comment/comfort features
        config: MatchingConfig object (uses defaults if None)
        
    Returns:
        Optimizer instance with results
    """
    if config is None:
        config = MatchingConfig()
    
    # ------------- Read and clean data -------------
    print(f"Loading data from {file_path} (sheet: {sheet})...")
    df = clean_dataframe(pd.read_excel(file_path, sheet_name=sheet))
    
    # Drop metadata columns (ID, timestamps, etc.) - keep only form data
    # Form data starts at column that contains "name" (the "Tell us your name" column)
    metadata_cols = ['ID', 'Hora de início', 'Hora de conclusão', 'Email', 'Nome', 'Hora da última modificação']
    cols_to_drop = [c for c in df.columns if any(m.lower() in c.lower() for m in metadata_cols)]
    if cols_to_drop:
        print(f"Dropping metadata columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Set index to name column (look for "Tell us" or "what's your name")
    name_col = find_column(df, "tell us") or find_column(df, "your name")
    if name_col:
        # Handle duplicate names by appending a suffix
        names = df[name_col].astype(str)
        seen = {}
        unique_names = []
        for name in names:
            if name in seen:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_names.append(name)
        df[name_col] = unique_names
        df = df.set_index(name_col)
        print(f"Using name column: {name_col}")
    else:
        print(f"Warning: Name column not found. Using first column: {df.columns[0]}")
        df = df.set_index(df.columns[0])
    
    print(f"Loaded {len(df)} participants")
    print(f"Remaining columns: {list(df.columns[:5])}...")
    
    # ------------- Extract comments and comfort preferences -------------
    comments = {}
    comfort = {}
    
    comment_col = find_column(df, "comment")
    if comment_col:
        comments = df[comment_col].to_dict()
    
    comfort_col = find_column(df, "comfortable")
    if comfort_col:
        comfort_values = df[comfort_col].apply(
            lambda x: str(x).strip().lower() == 'yes'
        )
        comfort = comfort_values.to_dict()
    
    # ------------- Split by type (buddy vs international) -------------
    print("Splitting data by participant type...")
    type_col = find_column(df, "I am a")
    if type_col:
        type_col_idx = df.columns.get_loc(type_col)
        print(f"Using type column: {type_col}")
    else:
        type_col_idx = 0
        print("Warning: Type column 'I am a' not found, using first column")
    df_groups = split_dfs(df, type_col_idx, sexo=sexo)
    
    # ------------- Create info dictionaries -------------
    infos = {}
    for group_name, df_group in df_groups.items():
        info = get_info_dict(df_group)
        
        # Add comments and comfort to each group
        group_names = list(df_group.index)
        info['comments'] = pd.Series([comments.get(n, "") for n in group_names], index=group_names)
        info['comfort'] = pd.Series([comfort.get(n, True) for n in group_names], index=group_names)
        
        infos[group_name] = info
    
    # Add field columns for preprocessing
    any_df = list(infos.keys())[0]
    infos['fields_cols'] = infos[any_df]['fields'].columns
    
    # Remove excluded columns from fields
    for group_name, info in infos.items():
        if group_name == 'fields_cols':
            continue
        for col in config.excluded_columns:
            if col in info['fields'].columns:
                info['fields'] = info['fields'].drop(columns=[col])
    
    # Update fields_cols after exclusion
    infos['fields_cols'] = infos[any_df]['fields'].columns
    
    print(f"Groups: {[k for k in infos.keys() if k != 'fields_cols']}")
    
    # ------------- Preprocess features -------------
    print("Preprocessing features (one-hot encoding)...")
    preprocessing(infos)
    
    # ------------- Create and run optimizer -------------
    print("\n" + "="*60)
    print("Running optimization...")
    print("="*60 + "\n")
    
    if use_improved:
        optimizer = ImprovedOptimizer(
            infos=infos, 
            h_m=sexo, 
            config=config,
            classes=classes
        )
    else:
        optimizer = Optimizer(infos=infos, h_m=sexo, classes=classes)
    
    optimizer.optimize()
    
    print("\n" + "-"*60 + "\n")
    
    # ------------- Save results -------------
    os.makedirs(save_path, exist_ok=True)
    
    if to_excel:
        results_path = os.path.join(save_path, f"resultados_{sexo}.xlsx")
        optimizer.results.to_excel(results_path, index=False)
        print(f"Results saved to {results_path}")
        
        # If using improved optimizer, also save group summary
        if use_improved and hasattr(optimizer, 'get_group_summary'):
            summary_path = os.path.join(save_path, f"group_summary_{sexo}.xlsx")
            optimizer.get_group_summary().to_excel(summary_path, index=False)
            print(f"Group summary saved to {summary_path}")
    
    if to_csv:
        results_path = os.path.join(save_path, f"resultados_{sexo}.csv")
        optimizer.results.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
    
    # Save similarity matrices
    cos_sim_dir = os.path.join(save_path, "cos_similarity")
    os.makedirs(cos_sim_dir, exist_ok=True)
    
    cos_path = os.path.join(cos_sim_dir, f"cosine_similarity_{sexo}.xlsx")
    optimizer.save_cosine_similarity(cos_path)
    print(f"Cosine similarity saved to {cos_path}")
    
    if use_improved and hasattr(optimizer, 'save_final_similarity'):
        final_path = os.path.join(cos_sim_dir, f"final_similarity_{sexo}.xlsx")
        optimizer.save_final_similarity(final_path)
        print(f"Final similarity (with bonuses) saved to {final_path}")
    
    return optimizer


if __name__ == "__main__":
    parser = ArgumentParser(description="Buddy Program Matching System")
    
    # Data arguments
    parser.add_argument("--xlsx-path", type=str, default="data/dados.xlsx", 
                        help="Path to the Excel file with data")
    parser.add_argument("--sheet", type=str, default="Sheet1", 
                        help="Sheet name in the Excel file")
    
    # Output arguments
    parser.add_argument("--to-csv", action="store_true", help="Save results to CSV")
    parser.add_argument("--to-excel", action="store_true", help="Save results to Excel")
    parser.add_argument("--save-path", type=str, default=".", 
                        help="Directory to save results")
    
    # Matching arguments
    parser.add_argument("--s", type=str, choices=['m', 'h'], required=True, 
                        help="Specify the gender (m=mulher, h=homem)")
    parser.add_argument("--classes", type=str, nargs='+', default=['buddy', 'int'], 
                        help="Specify the classes to use")
    
    # New arguments for improved matching
    parser.add_argument("--legacy", action="store_true", 
                        help="Use legacy optimizer (without comment/comfort features)")
    parser.add_argument("--comment-weight", type=float, default=0.1,
                        help="Weight for comment similarity bonus (default: 0.1)")
    parser.add_argument("--buddy-weight", type=float, default=0.7,
                        help="Weight for buddy-student similarity (default: 0.7)")
    parser.add_argument("--student-weight", type=float, default=0.3,
                        help="Weight for student-student similarity (default: 0.3)")
    parser.add_argument("--comfort-bonus", type=float, default=0.1,
                        help="Bonus when both comfortable with different (default: 0.1)")
    parser.add_argument("--comfort-penalty", type=float, default=0.1,
                        help="Penalty when uncomfortable with different (default: 0.1)")
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = MatchingConfig(
        comment_bonus_weight=args.comment_weight,
        buddy_student_weight=args.buddy_weight,
        student_student_weight=args.student_weight,
        similar_bonus=args.comfort_bonus,
        different_penalty=args.comfort_penalty
    )
    
    main(
        to_csv=args.to_csv, 
        to_excel=args.to_excel, 
        save_path=args.save_path, 
        sexo=args.s, 
        classes=args.classes, 
        file_path=args.xlsx_path, 
        sheet=args.sheet,
        use_improved=not args.legacy,
        config=config
    )
