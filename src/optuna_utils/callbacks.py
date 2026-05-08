"""
Callbacks for Optuna studies.
"""

def logging_callback(study, trial):
    """Logs trial results."""
    print(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}")
