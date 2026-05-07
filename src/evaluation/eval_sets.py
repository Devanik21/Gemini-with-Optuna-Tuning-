"""
Evaluation datasets for prompt tuning.
"""

def get_summarization_dataset():
    """Returns a mock dataset for summarization."""
    return [
        {"input": "The quick brown fox jumps over the lazy dog.", "target": "A fox jumps over a dog."},
        {"input": "Artificial intelligence is revolutionizing the tech industry.", "target": "AI transforms tech."}
    ]
