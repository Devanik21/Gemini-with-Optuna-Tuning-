"""
Script to export the best configuration to a file.
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--output", type=str, default="config.json")
    args = parser.parse_args()
    print(f"Exporting config for {args.study_name} to {args.output}")

if __name__ == "__main__":
    main()
