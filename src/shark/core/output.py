"""Collision-safe, local-date analysis directories."""
from datetime import date
import os
from pathlib import Path


def create_job_directory(root: str | Path | None = None) -> Path:
    """Reserve a new job atomically; never reuse an existing analysis folder."""
    parent = Path(root or os.environ.get('SHARK_OUTPUT_DIR') or Path.cwd()).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    base = 'shark_job_' + date.today().strftime('%m%d%y')
    number = 1
    while True:
        path = parent / (base if number == 1 else f'{base}_{number}')
        try:
            path.mkdir()
        except FileExistsError:
            number += 1
            continue
        print(f'[OUTPUT] {path}: analysis inputs, results and report; keep for reproducibility, delete when no longer needed')
        return path
