from classes.optimizer import Optimizer
from utils import *
from argparse import ArgumentParser
import os


def main(to_csv: bool = False, to_excel: bool = False, save_path: str = ".", sexo: str = "m", classes: list[str] = ['buddy', 'int'], 
         file_path: str = "data/dados.xlsx", sheet: str = "Sheet1") -> None:

    # ------------- lendo e limpando os dados -------------

    # DataFrame com os dados do Excel !!!DEVE CONTER A COLUNA "NAME"!!!
    df = clean_dataframe(pd.read_excel(file_path, sheet_name=sheet)).set_index('Name')


    # ------------- separando os dados em grupos de acordo com o gênero e tipo -------------
    """{
        {sexo}_{class0} : df_cls0 , 
        {sexo}_{class1} : df_cls1, 
        ... , 
        {sexo}_{classN} : df_clsN
    }"""
    df_groups = split_dfs(df, 0, sexo=sexo)
    # --------------------------------------------------------------------------------------

    # ------------- criando dicionários com informações de cada grupo -------------

    # dicionario com informações de cada grupo
    # ver documentação de get_info_dict para mais detalhes
    infos = {g : get_info_dict(df_group) for g, df_group in df_groups.items()}

    # adicionando nomes das colunas para facilitar o acesso da informação
    any_df = list(infos.keys())[0]
    infos['fields_cols'] = infos[any_df]['fields'].columns
    # ------------------------------------------------


    # ------------- preprocessando os dados -------------

    # Transforma as colunas de texto em vetores numéricos
    preprocessing(infos)

    # ------------- criando e rodando o otimizador -------------

    optimizer : Optimizer = Optimizer(infos=infos, h_m=sexo, classes=classes)
    optimizer.optimize()
    print("\n--------------------------------------------------------------\n")

    # ------------- salvando os resultados em arquivos -------------
    if to_excel:
        optimizer.results.to_excel(f"{save_path}/resultados_{sexo}.xlsx")        
    if to_csv:
        optimizer.results.to_csv(f"{save_path}/resultados_{sexo}.csv")

    path = os.path.join(save_path, "cos_similarity", f"cosine_similarity_{sexo}.xlsx")
    optimizer.save_cosine_similarity(path)

    print(f"Cosine similarity saved to {path}")

    return optimizer

if __name__ == "__main__":
    parser = ArgumentParser(description="Process some integers.")
    parser.add_argument("--xlsx-path", type=str, default="data/dados.xlsx", help="Path to the Excel file with data")
    parser.add_argument("--sheet", type=str, default="Sheet1", help="Sheet name in the Excel file")
    parser.add_argument("--to-csv", action="store_true", help="Save results to CSV")
    parser.add_argument("--to-excel", action="store_true", help="Save results to Excel")
    parser.add_argument("--save-path", type=str, default=".", help="Directory to save results")
    parser.add_argument("--s", type=str, choices=['m', 'h'], required=True, help="Specify the gender (m or h)")
    parser.add_argument("--classes", type=str, nargs='+', default=['buddy', 'int'], help="Specify the classes to use")
    args = parser.parse_args()

    main(to_csv=args.to_csv, to_excel=args.to_excel, save_path=args.save_path, sexo=args.s, classes=args.classes, file_path=args.xlsx_path, sheet=args.sheet)