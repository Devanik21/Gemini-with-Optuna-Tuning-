"""
Main script to run an Optuna study.
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="summarisation")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--study_name", type=str, default="gemini_study")
    args = parser.parse_args()
    print(f"Running study {args.study_name} for task {args.task} with {args.trials} trials.")

if __name__ == "__main__":
    main()
