from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus

@dataclass
class Optimizer:
    infos: Dict # Dicionário contendo as informações dos buddies e intercambistas
    h_m: str  # homens ou mulheres
    classes: List[str] = field(default_factory=lambda: ['buddy', 'int'])  # Lista de classes para a separação dos dados

    def __post_init__(self):
        ks = list(self.infos.keys())
        self.buddies: pd.DataFrame = self.infos[ks[0]]['fields']
        self.international: pd.DataFrame = self.infos[ks[1]]['fields']
        self.cos_similarity: pd.DataFrame = self.__get_cosine_similarity()
        self.ratio: float = self.__get_ratio()

        # Define os limites mínimo e máximo de intercambistas por buddy
        min_bound, max_bound = self.__define_bounds()
        self.min_bound: int = min_bound
        self.max_bound: int = max_bound

        # Define as variáveis do problema de otimização
        # Exemplo: {'x_h_buddy': {(0, 0): var1, (0, 1): var2, ...}, ...}
        self.var_x: Dict[str, Dict[tuple, LpVariable]] = self.__define_var()
        self.problem: LpProblem = self.__create_problem()
        self.results: pd.DataFrame = None

    def __get_ratio(self) -> float:
        """
        Returns the ratio of international students per buddy.
        """
        return self.cos_similarity.shape[1] / self.cos_similarity.shape[0]
    

    def __get_cosine_similarity(self) -> pd.DataFrame:
        """
        Returns the cosine similarity DataFrame.
        """
        return pd.DataFrame(cosine_similarity(self.buddies, self.international),
                        index=self.buddies.index,
                        columns=self.international.index)


    def __define_bounds(self) -> Tuple[int, int]:
        """
        Define the minimum and maximum number of international students per buddy.
        """
        min_bound = int(self.ratio)
        max_bound = min_bound + 1 if self.ratio % 1 != 0 else min_bound
        return min_bound, max_bound

    def __define_var(self) -> Dict[str, Dict[Tuple[int, int], LpVariable]]:
        """
        Define the binary variables for the optimization problem.
        """
        return LpVariable.dicts(
            f"x_{self.h_m}",
            ((i, j) for i in self.buddies.index for j in self.international.index),
            cat='Binary'
        )

    def __create_problem(self) -> LpProblem:
        """
        Create the optimization problem.
        """
        prob_m = LpProblem(f"Matching_{self.h_m}", LpMaximize)

        # Restricao: cada intercambista recebe exatamente um buddy
        for j in self.international.index:
            prob_m += lpSum(self.var_x[(i, j)] for i in self.buddies.index) == 1, f"OneBuddyPer{self.h_m.capitalize()}_{j}"

        # Restricao de capacidade de cada buddy (minimo e maximo)
        for i in self.buddies.index:
            prob_m += lpSum(self.var_x[(i, j)] for j in self.international.index) >= self.min_bound, f"MinPerBuddy{self.h_m.capitalize()}_{i}"
            prob_m += lpSum(self.var_x[(i, j)] for j in self.international.index) <= self.max_bound, f"MaxPerBuddy{self.h_m.capitalize()}_{i}"

        # Funcao objetivo: maximizar similaridade cosseno
        prob_m += lpSum(
            self.cos_similarity.loc[i, j] * self.var_x[(i, j)]
            for i in self.buddies.index
            for j in self.international.index
        ), f"MaximizeCosineSim{self.h_m.capitalize()}"

        return prob_m

    def __get_results__(self) -> pd.DataFrame:
        # Extrair solucao
        assignments_m = []
        for (i, j), var in self.var_x.items():
            if var.value() == 1:
                assignments_m.append((i, j, self.cos_similarity.loc[i, j]))

        # return pd.DataFrame(assignments_m, columns=['Buddy_Female', 'Intercambista_Female', 'Score'])
        return pd.DataFrame(assignments_m, columns=[f"{cls.capitalize()}_{self.h_m.capitalize()}" for cls in self.classes] + ["Cosine_Similarity"])


    def optimize(self) -> None:
        """
        Solve the optimization problem.
        """
        self.problem.solve()
        if self.problem.status == 1:
            print(f"Status: {self.problem.status}, {LpStatus[self.problem.status]}\n")
        else:
            print(f"Status: {self.problem.status}, {LpStatus[self.problem.status]}")
            raise Exception("Optimization problem could not be solved successfully.")

        self.results = self.__get_results__()

        
    def save_cosine_similarity(self, filename: str) -> None:
        """
        Save the cosine similarity DataFrame to an Excel file.
        """
        self.cos_similarity.to_excel(filename)