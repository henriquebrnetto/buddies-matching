"""
Person model for the Buddy Program matching system.
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Person:
    """Represents a participant in the buddy program."""
    
    name: str
    person_type: str  # "buddy" or "international"
    contact: str
    comment: str
    comfortable_with_different: bool
    raw_features: pd.Series = field(repr=False)  # Original feature values
    processed_features: Optional[pd.Series] = field(default=None, repr=False)  # One-hot encoded
    
    @property
    def is_buddy(self) -> bool:
        return self.person_type == "buddy"
    
    @property
    def is_international(self) -> bool:
        return self.person_type == "international"
    
    @property
    def has_comment(self) -> bool:
        """Returns True if person has a meaningful comment."""
        if not self.comment or pd.isna(self.comment):
            return False
        # Filter trivial comments
        trivial = {'.', '/', '-', 'nan', '', 'n/a', 'na'}
        return self.comment.strip().lower() not in trivial
