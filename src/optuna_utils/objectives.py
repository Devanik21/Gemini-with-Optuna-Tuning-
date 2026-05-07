"""
Optuna objective functions for hyperparameter tuning.
"""

def create_objective(client, dataset, metric_fn, task_type):
    """Creates an objective function for Optuna."""
    def objective(trial):
        # Hyperparameters
        variant = trial.suggest_int("prompt_variant", 0, 2)
        temperature = trial.suggest_float("temperature", 0.0, 1.0)

        scores = []
        from src.gemini.prompts import get_prompt
        for item in dataset:
            prompt = get_prompt(task_type, variant, item["input"])
            prediction = "Mock prediction" # MOCK for testing
            score = metric_fn(prediction, item["target"])
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0
    return objective
