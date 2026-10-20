"""Small, dependency-free helpers shared by isolated Flow W8 experiments."""
import hashlib
from pathlib import Path

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
