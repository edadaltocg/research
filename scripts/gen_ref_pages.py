# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate the code reference pages."""

import os
from pathlib import Path

root = Path(__file__).parent.parent
src = root / "research"
prefix = "research"
ref_dir = root / "docs" / "reference"

# Create docs/reference directory if it doesn't exist
ref_dir.mkdir(parents=True, exist_ok=True)

# Clear existing files in docs/reference
for path in list(ref_dir.glob("**/*")):
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass

print(f"Generating reference pages from {src}")

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")

    full_doc_path = ref_dir / doc_path

    parts = tuple(module_path.parts)
    parts = (prefix, *parts)

    if parts[-1] == "__init__" or parts[-1] == "__main__":
        continue

    # Ensure the parent directory exists
    full_doc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating reference page for {path} -> {full_doc_path}")
    with open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)
