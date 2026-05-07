"""
Global settings and configuration.
"""
import os

class Settings:
    DEFAULT_TASK = os.getenv("DEFAULT_TASK", "summarisation")
    N_TRIALS = int(os.getenv("N_TRIALS", "100"))
    STUDY_DB = os.getenv("STUDY_DB", "studies/optuna_study.db")

settings = Settings()
