"""
Script to extract the best configuration from a completed study.
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True)
    args = parser.parse_args()
    print(f"Extracting best config for {args.study_name}")

if __name__ == "__main__":
    main()
