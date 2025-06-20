import contextlib
import tempfile
from pathlib import Path
from typing import Generator


@contextlib.contextmanager
def temp_dir_if_none(dir_path: Path | None) -> Generator[Path, None, None]:
    if dir_path is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        yield dir_path
