"""
Prompt templates and generation utilities.
"""

PROMPT_TEMPLATES = {
    "summarization": [
        "Summarize the following text: {text}",
        "Provide a brief summary of: {text}",
        "TL;DR: {text}"
    ],
    "classification": [
        "Classify the sentiment of this text: {text}",
        "Is the following text positive or negative? {text}"
    ]
}

def get_prompt(task: str, variant: int, text: str) -> str:
    """Returns a formatted prompt."""
    templates = PROMPT_TEMPLATES.get(task, ["{text}"])
    template = templates[variant % len(templates)]
    return template.format(text=text)
