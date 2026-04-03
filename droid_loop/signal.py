import json
from pathlib import Path

from .catalog import load_moments


def export(catalog_path: Path, output_path: Path) -> None:
    """Export cataloged moments to a training-ready JSON file."""
    moments = list(load_moments(catalog_path))
    output_path.write_text(json.dumps(moments, indent=2))
    print(f"Exported {len(moments)} moments -> {output_path}")
