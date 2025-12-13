import re
import json
from pathlib import Path


def normalize_words(text: str) -> list[str]:
    """Extract and normalize words from text."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if 3 <= len(w) <= 12]

## RUN NOTEBOOK TO GET DICTIONARY
_notebooks_dir = Path(__file__).parent.parent / "notebooks"
_dialect_dict_path = _notebooks_dir / "dialect_dict.json"

with open(_dialect_dict_path, 'r') as f:
    _dialect_dict = json.load(f)
    
_dialect_dict_sets = {k: set(v) for k, v in _dialect_dict.items()}


def detect_dialect(text: str) -> str:
    """
    Detect dialect from text using word matching.
    
    Args:
        text: Input text to classify
        
    Returns:
        Detected dialect name or "Unknown" if no match found
    """
    if not _dialect_dict_sets:
        return "Unknown"
    
    words = set(normalize_words(text))
    
    max_hits = 0
    max_dialect = None
    
    for dialect, dict_words in _dialect_dict_sets.items():
        hits = len(words & dict_words)
        if hits > max_hits:
            max_hits = hits
            max_dialect = dialect
    
    if max_hits > 0 and max_dialect is not None:
        return max_dialect
    return "Unknown"
