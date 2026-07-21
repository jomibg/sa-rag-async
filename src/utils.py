from typing import Dict, Any
import json

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with consistent error handling.

    Args:
        file_path: Path to the JSON file to load

    Returns:
        Parsed JSON content as dictionary

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If JSON is malformed
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {file_path}: {e}")


def save_json_file(file_path: str, data: Any) -> None:
    """
    Save data to JSON file.

    Args:
        file_path: Path where to save the JSON file
        data: Data to serialize as JSON
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)