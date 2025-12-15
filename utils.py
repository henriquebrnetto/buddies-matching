import pandas as pd
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import re

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a Pandas DataFrame by removing the '\xa0' (non-breaking space)
    character from all string values within the DataFrame and from all column names.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: A new DataFrame with '\xa0' characters removed from
                      string data and column names.
    """
    df_cleaned = df.copy()

    new_columns = [col.strip() for col in df_cleaned.columns]
    df_cleaned.columns = new_columns

    for col in df_cleaned.select_dtypes(include=['object']).columns:
        p = re.compile(r'\xa0')
        # Remove non-breaking spaces from string values
        col_cleaned = df_cleaned[col].astype(str).str.strip()

        df_cleaned[col] = col_cleaned.apply(lambda x: p.sub('', x) if isinstance(x, str) else x)

    return df_cleaned


def is_list(obj : pd.Series) -> bool:
    return len(obj.values[0].split(';')) > 1


def get_info_dict(df : pd.DataFrame, names_col:str=None) -> dict:
    """
    Extracts relevant information from a DataFrame and returns it as a dictionary.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing buddy information.
    Returns:
        dict: A dictionary containing the names, contacts, and fields of the buddies. \
        {
        'names' : Series of names,
        'contacts' : Series of contacts,
        'fields' : DataFrame of fields with one-hot encoding.
        }
    """
    return {
        'names' : df.index if names_col is None else df.loc[:, names_col],
        'contacts' : df.iloc[:, -2],
        'fields' : df.iloc[:, 2:-3]
        }


def get_ohe_categories(infos : dict, col: str) -> pd.DataFrame:
    """
        Get the categories from the OneHotEncoder for all fields in infos.
        
        Args:
            infos (dict): A dictionary containing information about different fields.
            column (str): The column name for which to retrieve the categories.
        Returns:
            pd.DataFrame: A DataFrame containing all categories for the specified column.
    """
    # Extract the values for the specified column from each info dictionary
    vals = pd.concat([info['fields'] for name, info in infos.items()
                      if name.startswith('h_') or name.startswith('m_')], 
                      ignore_index=True)
    
    vals = vals[col].dropna().unique()
    return pd.DataFrame(vals, columns=[col]).fillna('')


def explode_col(df : pd.DataFrame, col : str, idx=None, sep=";", dummies:set=None) -> pd.DataFrame:
    """
        Explode a column in a DataFrame that contains lists.
        Args:
            df (pd.DataFrame): The input DataFrame.
            col (str): The name of the column to explode.
            idx (pd.Series, optional): The index to use for the 'Name' column. Defaults to None.
            sep (str, optional): The separator used to split the column values. Defaults to ";".
        Returns:
            pd.DataFrame: A DataFrame with the 'Name' column and one-hot encoded columns as 'columnName_value'.
            
    """

    df : pd.Series = df[col].copy()
    
    # Clean the index names - remove \xa0 and other whitespace
    if idx:
        clean_idx = idx.astype(str).str.replace('\xa0', '').str.strip()
        df.set_index(clean_idx, drop=True, inplace=True)

    # Clean the column data and explode it
    df = df.astype(str).str.replace('\xa0', '').str.strip().str.split(sep)
    df_new = df.explode()

    df_new = df_new[(df_new != '')].dropna()

    result = pd.get_dummies(df_new, prefix=col).groupby(df_new.index).sum()

    if dummies is not None:
        dummies.update(result.columns)

    return result


def simple_string_ohe(infos: dict, df : pd.DataFrame, cols: list, dummies: set=None) -> pd.DataFrame:

    dfs = []

    for col in cols:
        
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        ohe.fit(get_ohe_categories(infos, col=col))
        temp = pd.DataFrame(ohe.transform(df[col].to_frame()),
                             columns=ohe.get_feature_names_out(),
                             index=df.index)

        if dummies is not None:
            dummies.update(temp.columns)
            
        dfs.append(temp)

    return pd.concat(dfs, axis=1)


def preprocessing(infos: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """    
    Preprocesses the data in the provided dictionary of information.
    This function handles the following tasks:
        - Explodes columns that contain lists into separate columns.
        - Applies OneHotEncoding to columns with simple string values.
        - Merges the processed data back into the original DataFrame.
    Args:
        infos (dict): A dictionary containing information about different groups.
                      {'names' : Series of names,
                        'contacts' : Series of contacts,
                        'fields' : DataFrame of fields with one-hot encoding.}
    Returns:
        None: The function modifies the 'fields' DataFrame in place for each group.
    """
    
    unique_values_lists = set()
    unique_values_simple = set()

    
    # infos = {
    #     {sexo}_{class0} : df_cls0 , 
    #     {sexo}_{class1} : df_cls1, 
    #     ... , 
    #     {sexo}_{classN} : df_clsN
    # }
    print(infos.keys())
    for group_name, info in infos.items():
        print(f'Processing group: {group_name}')

        # Verifica se o nome do grupo começa com 'h_' ou 'm_'
        if not (group_name.startswith('h_') or group_name.startswith('m_')):
            continue

        # Pega nomes das colunas de valor simples (str)
        simple_cols = []

        # Pega DataFrames explodidos
        exploded_dfs = []

        # Cria cópia dos dados do grupo
        info_data = info['fields'].copy()

        # Iremos processar todas as colunas
        for col in infos['fields_cols']:


            if is_list(info['fields'][col]):

                # DataFrame com os nomes e valores da coluna explodidos
                exploded_dfs.append(explode_col(info_data, col, sep=';', dummies=unique_values_lists))

            # Caso a coluna não seja uma lista, iremos aplicar o OneHotEncoder
            else:

                # Coluna com valores simples (não listas)
                simple_cols.append(col)
                

        # Processa as colunas que não são listas
        ohe_cols = simple_string_ohe(infos, info_data, simple_cols)

        # Concatena os DataFrames explodidos
        exploded_df = pd.concat(exploded_dfs, axis=1)

        # Concatena os DataFrames explodidos com os OneHotEncoded
        info_data = pd.concat([info_data, exploded_df, ohe_cols], axis=1)

        # Remove as colunas originais que foram processadas
        info_data.drop(columns=infos['fields_cols'].copy(), inplace=True)

        info['fields'] = info_data
    
    # Cria um conjunto de colunas únicas a partir dos valores únicos encontrados
    final_cols = unique_values_lists | unique_values_simple
    for group_name, info in infos.items():

        # Verifica se o nome do grupo começa com 'h_' ou 'm_'
        if not (group_name.startswith('h_') or group_name.startswith('m_')):
            continue

        info_data = info['fields'].copy()
        cols_to_add = final_cols - set(info_data.columns)
        if cols_to_add:
            info_data[list(cols_to_add)] = 0
        
        info_data = info_data[list(final_cols)]
        info['fields'] = info_data


def split_dfs(df: pd.DataFrame, col:int, sexo: str="") -> Dict[str, pd.DataFrame]:
    """
    Splits a DataFrame into two DataFrames based on the unique values in a specified column.
    """

    return {f"{sexo}_{opt}": v for opt, v in df.groupby(df.columns[col])}