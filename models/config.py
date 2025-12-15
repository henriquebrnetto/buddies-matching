"""
Configuration classes for the Buddy Program matching system.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchingConfig:
    """Configuration for the matching optimization."""
    
    # Column configuration - either list of column names or (start, end) indices
    matching_columns: list[str] | tuple[int, int] = field(default_factory=lambda: (8, 18))
    
    # Column names for special fields
    name_column: str = "Tell us what's you name: "
    type_column: str = "I am a: "
    contact_column: str = "Would you share your instagram @ and whats app number so we can share with your buddy? If yes, please, write It bellow:"
    comment_column: str = "Comments"
    comfort_column: str = "Would you be comfortable being matched with someone very different from you (personality, habits, culture)?"
    
    # Type identifiers
    buddy_identifier: str = "Brazilian student (Buddy)"
    international_identifier: str = "International student (Incoming)"
    
    # Weight configuration (all matching columns equal weight, only comment bonus configurable)
    comment_bonus_weight: float = 0.1
    
    # Optimization weights
    buddy_student_weight: float = 0.7
    student_student_weight: float = 0.3
    
    # Comfort modifiers (for "Would you be comfortable..." column)
    similar_bonus: float = 0.1   # Bonus when both prefer similar people
    different_penalty: float = 0.1  # Penalty when "No" paired with very different person
    
    # Columns to exclude from feature vector (used as modifiers instead)
    excluded_columns: list[str] = field(default_factory=lambda: [
        "Would you be comfortable being matched with someone very different from you (personality, habits, culture)?"
    ])
    
    def get_matching_column_range(self) -> tuple[int, int]:
        """Returns (start, end) indices for matching columns."""
        if isinstance(self.matching_columns, tuple):
            return self.matching_columns
        raise ValueError("When using column names, use get_matching_column_names() instead")
    
    def get_matching_column_names(self) -> list[str]:
        """Returns list of column names for matching."""
        if isinstance(self.matching_columns, list):
            return self.matching_columns
        raise ValueError("When using indices, use get_matching_column_range() instead")
    
    def uses_column_indices(self) -> bool:
        """Returns True if using column indices, False if using column names."""
        return isinstance(self.matching_columns, tuple)
