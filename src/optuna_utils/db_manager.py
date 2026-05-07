"""
Database management for Optuna studies.
"""
import os

def get_db_url(study_name: str) -> str:
    """Returns the SQLite URL for the study database."""
    db_path = os.path.join("studies", f"{study_name}.db")
    return f"sqlite:///{db_path}"
