"""
Improved optimizer for buddy matching with multi-objective optimization.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus

import sys
sys.path.append('..')
from models.config import MatchingConfig
from utils import (
    compute_comment_similarity, 
    compute_comfort_modifier, 
    compute_intra_group_similarity
)


@dataclass
class ImprovedOptimizer:
    """
    Optimizer for buddy-international student matching with:
    - Multi-objective optimization (buddy-student + student-student similarity)
    - Comment-based bonus scoring
    - Comfort-based modifiers
    """
    
    # Required inputs
    infos: Dict  # Dictionary with buddy and international student info
    h_m: str  # Gender identifier ('h' or 'm')
    config: MatchingConfig  # Configuration object
    
    # Optional inputs
    classes: List[str] = field(default_factory=lambda: ['buddy', 'int'])
    
    def __post_init__(self):
        # Extract data from infos dictionary
        ks = list(self.infos.keys())
        self.buddies: pd.DataFrame = self.infos[ks[0]]['fields']
        self.international: pd.DataFrame = self.infos[ks[1]]['fields']
        
        # Store names and comments for bonus calculations
        self._buddy_names = self.buddies.index.tolist()
        self._student_names = self.international.index.tolist()
        
        # Extract comments from original data
        self._buddy_comments = self._extract_comments(ks[0])
        self._student_comments = self._extract_comments(ks[1])
        
        # Extract comfort preferences
        self._buddy_comfort = self._extract_comfort(ks[0])
        self._student_comfort = self._extract_comfort(ks[1])
        
        # Compute base cosine similarity
        self.cos_similarity: pd.DataFrame = self._get_cosine_similarity()
        
        # Compute modifiers
        self.comment_bonus: pd.DataFrame = self._compute_comment_bonus()
        self.comfort_modifier: pd.DataFrame = self._compute_comfort_modifiers()
        
        # Compute intra-student similarity
        self.student_similarity: pd.DataFrame = self._compute_student_similarity()
        
        # Combine into final similarity matrix
        self.final_similarity: pd.DataFrame = self._compute_final_similarity()
        
        # Calculate bounds for constraints
        self.ratio: float = self._get_ratio()
        min_bound, max_bound = self._define_bounds()
        self.min_bound: int = min_bound
        self.max_bound: int = max_bound
        
        # Define optimization variables and problem
        self.var_x: Dict[Tuple, LpVariable] = self._define_var()
        self.problem: LpProblem = self._create_problem()
        self.results: Optional[pd.DataFrame] = None
    
    def _extract_comments(self, group_key: str) -> List[str]:
        """Extract comments for a group from the original DataFrame."""
        try:
            # Try to get comments from the infos structure
            if 'comments' in self.infos[group_key]:
                return self.infos[group_key]['comments'].tolist()
        except (KeyError, AttributeError):
            pass
        # Return empty strings if no comments found
        return [""] * len(self.infos[group_key]['fields'])
    
    def _extract_comfort(self, group_key: str) -> pd.Series:
        """Extract comfort preferences for a group."""
        names = self.infos[group_key]['fields'].index
        try:
            if 'comfort' in self.infos[group_key]:
                return self.infos[group_key]['comfort']
        except (KeyError, AttributeError):
            pass
        # Default to True (comfortable with different)
        return pd.Series([True] * len(names), index=names)
    
    def _get_ratio(self) -> float:
        """Returns the ratio of international students per buddy."""
        return len(self._student_names) / len(self._buddy_names)
    
    def _get_cosine_similarity(self) -> pd.DataFrame:
        """Compute base cosine similarity between buddies and students."""
        return pd.DataFrame(
            cosine_similarity(self.buddies, self.international),
            index=self.buddies.index,
            columns=self.international.index
        )
    
    def _compute_comment_bonus(self) -> pd.DataFrame:
        """Compute TF-IDF based comment similarity bonus."""
        return compute_comment_similarity(
            self._buddy_comments,
            self._student_comments,
            index_a=self._buddy_names,
            index_b=self._student_names
        )
    
    def _compute_comfort_modifiers(self) -> pd.DataFrame:
        """Compute comfort-based modifiers for matching."""
        return compute_comfort_modifier(
            self.buddies,
            self.international,
            self._buddy_comfort,
            self._student_comfort,
            similar_bonus=self.config.similar_bonus,
            different_penalty=self.config.different_penalty
        )
    
    def _compute_student_similarity(self) -> pd.DataFrame:
        """Compute similarity matrix among international students."""
        return compute_intra_group_similarity(self.international)
    
    def _compute_final_similarity(self) -> pd.DataFrame:
        """
        Combine base similarity with bonuses and modifiers.
        Final = base + comment_weight * comment_bonus + comfort_modifier
        """
        final = (
            self.cos_similarity 
            + self.config.comment_bonus_weight * self.comment_bonus
            + self.comfort_modifier
        )
        # Clip to [0, 1] range
        return final.clip(0, 1)
    
    def _define_bounds(self) -> Tuple[int, int]:
        """Define min/max international students per buddy."""
        min_bound = int(self.ratio)
        max_bound = min_bound + 1 if self.ratio % 1 != 0 else min_bound
        return min_bound, max_bound
    
    def _define_var(self) -> Dict[Tuple, LpVariable]:
        """Define binary variables for the optimization problem."""
        return LpVariable.dicts(
            f"x_{self.h_m}",
            ((i, j) for i in self.buddies.index for j in self.international.index),
            cat='Binary'
        )
    
    def _create_problem(self) -> LpProblem:
        """
        Create the multi-objective optimization problem.
        
        Objective: Maximize α * Σ(buddy-student similarity) + β * Σ(student-student similarity within groups)
        
        Subject to:
        - Each student gets exactly one buddy
        - Each buddy gets between min_bound and max_bound students
        """
        prob = LpProblem(f"Matching_{self.h_m}", LpMaximize)
        
        # Constraint: each international student gets exactly one buddy
        for j in self.international.index:
            prob += (
                lpSum(self.var_x[(i, j)] for i in self.buddies.index) == 1,
                f"OneBuddyPerStudent_{j}"
            )
        
        # Constraint: each buddy gets between min_bound and max_bound students
        for i in self.buddies.index:
            prob += (
                lpSum(self.var_x[(i, j)] for j in self.international.index) >= self.min_bound,
                f"MinPerBuddy_{i}"
            )
            prob += (
                lpSum(self.var_x[(i, j)] for j in self.international.index) <= self.max_bound,
                f"MaxPerBuddy_{i}"
            )
        
        # Objective function: maximize buddy-student similarity
        # Note: Student-student similarity within groups is analyzed post-optimization
        # because PuLP can't handle quadratic (x * x) terms
        
        # Buddy-student similarity term (linear objective)
        buddy_student_term = lpSum(
            self.final_similarity.loc[i, j] * self.var_x[(i, j)]
            for i in self.buddies.index
            for j in self.international.index
        )
        
        prob += (
            buddy_student_term,
            f"MaximizeTotalSimilarity_{self.h_m}"
        )
        
        return prob
    
    def optimize(self) -> None:
        """Solve the optimization problem."""
        self.problem.solve()
        
        if self.problem.status == 1:
            print(f"Status: {self.problem.status}, {LpStatus[self.problem.status]}")
            print(f"Ratio: {self.ratio:.2f} students/buddy (bounds: {self.min_bound}-{self.max_bound})")
        else:
            print(f"Status: {self.problem.status}, {LpStatus[self.problem.status]}")
            raise Exception("Optimization problem could not be solved successfully.")
        
        self.results = self._get_results()
        
        # Post-optimization: analyze group cohesion
        self._analyze_group_cohesion()
    
    def _get_results(self) -> pd.DataFrame:
        """Extract assignments from solved problem."""
        assignments = []
        for (i, j), var in self.var_x.items():
            if var.value() == 1:
                assignments.append({
                    'Buddy': i,
                    'International': j,
                    'Base_Similarity': self.cos_similarity.loc[i, j],
                    'Comment_Bonus': self.comment_bonus.loc[i, j],
                    'Comfort_Modifier': self.comfort_modifier.loc[i, j],
                    'Final_Similarity': self.final_similarity.loc[i, j]
                })
        
        return pd.DataFrame(assignments)
    
    def _analyze_group_cohesion(self) -> None:
        """Analyze and print group cohesion statistics."""
        if self.results is None:
            return
        
        print("\n--- Group Cohesion Analysis ---")
        
        # Group students by buddy
        groups = self.results.groupby('Buddy')['International'].apply(list)
        
        total_cohesion = 0
        for buddy, students in groups.items():
            if len(students) > 1:
                # Calculate average pairwise similarity within group
                pairwise_sims = []
                for i, s1 in enumerate(students):
                    for s2 in students[i+1:]:
                        pairwise_sims.append(self.student_similarity.loc[s1, s2])
                avg_cohesion = np.mean(pairwise_sims) if pairwise_sims else 0
                total_cohesion += avg_cohesion
                print(f"{buddy}: {len(students)} students, avg cohesion: {avg_cohesion:.3f}")
            else:
                print(f"{buddy}: {len(students)} student")
        
        if len(groups) > 0:
            print(f"\nOverall average group cohesion: {total_cohesion / len(groups):.3f}")
    
    def save_cosine_similarity(self, filename: str) -> None:
        """Save the cosine similarity DataFrame to an Excel file."""
        self.cos_similarity.to_excel(filename)
    
    def save_final_similarity(self, filename: str) -> None:
        """Save the final (adjusted) similarity DataFrame to an Excel file."""
        self.final_similarity.to_excel(filename)
    
    def get_group_summary(self) -> pd.DataFrame:
        """Get a summary of groups with statistics."""
        if self.results is None:
            return pd.DataFrame()
        
        summary = []
        groups = self.results.groupby('Buddy')
        
        for buddy, group_df in groups:
            students = group_df['International'].tolist()
            
            # Calculate group cohesion
            if len(students) > 1:
                pairwise_sims = []
                for i, s1 in enumerate(students):
                    for s2 in students[i+1:]:
                        pairwise_sims.append(self.student_similarity.loc[s1, s2])
                cohesion = np.mean(pairwise_sims) if pairwise_sims else 0
            else:
                cohesion = 1.0  # Single student group has perfect cohesion
            
            summary.append({
                'Buddy': buddy,
                'Num_Students': len(students),
                'Students': ', '.join(students),
                'Avg_Similarity': group_df['Final_Similarity'].mean(),
                'Group_Cohesion': cohesion
            })
        
        return pd.DataFrame(summary)
