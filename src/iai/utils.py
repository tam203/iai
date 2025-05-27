import datetime
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def generate_run_id() -> str:
    """Generates a unique run ID based on timestamp and a short UUID."""
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{now}_{short_uuid}"

def get_run_data_path(run_id: str, ensure_exists: bool = True) -> Path:
    """
    Gets the path to the data directory for a specific run.
    Creates the directory if it doesn't exist and ensure_exists is True.
    """
    run_path = DATA_DIR / run_id
    if ensure_exists:
        run_path.mkdir(parents=True, exist_ok=True)
    return run_path

def get_filepath_in_run_data(run_id: str, filename: str, ensure_run_dir_exists: bool = True) -> Path:
    """Constructs a full file path within a specific run's data directory."""
    run_data_path = get_run_data_path(run_id, ensure_exists=ensure_run_dir_exists)
    return run_data_path / filename