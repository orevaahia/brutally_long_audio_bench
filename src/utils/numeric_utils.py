import ast
import json
import re
import numpy as np


def load_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data has been saved to {file_path}")


def remove_bracketed_tags(text):
    # For Phi
    # This will remove anything like <|something|>
    return re.sub(r'<\|.*?\|>', '', text)


def extract_integer(text):
    """Extracts the first integer found in a given text."""
    match = re.search(r'\b\d+\b', str(text))  # Find first standalone integer
    return int(match.group()) if match else None  # Convert to int if found

def convert_to_seconds(value):
    if isinstance(value, str):
        if ':' in value:  # MM:SS format
            minutes, seconds = map(float, value.split(':'))
            return minutes * 60 + seconds
        elif '.' in value:  # MM.SS format
            minutes, seconds = map(float, value.split('.'))
            return minutes * 60 + seconds
        else:
            return int(value)
    try:
        return float(value)  # Keep numeric values unchanged
    except ValueError:
        return None  # Handle unexpected cases

def convert_minutes_to_seconds(value):
    if isinstance(value, str):
        if ':' in value:  # MM:SS format
            minutes, seconds = map(float, value.split(':'))
            return int(minutes * 60 + seconds)
    try:
        return int(float(value) * 60)  # Convert minutes to seconds
    except ValueError:
        return None  # Handle unexpected cases


def safe_literal_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x

def is_correct(pred, gt):
    # If the ground truth (gt) is a range (list), checks if the prediction (pred) falls within it.
    # (This only applies to the speaker number estimation task.)
    # Otherwise, checks for exact match.

    if isinstance(gt, list):  # Check if ground truth is a range
        return gt[0] <= pred <= gt[1]
    else:  # Otherwise, it's a single number
        return gt == pred

def is_correct_threshold_offset(pred, gt, offset, threshold=0):
    """
    Check if the predicted value (pred) falls within an tolerance range of the ground truth (gt),
    considering a given offset only if gt is above a threshold.
    """
    def apply_offset(value):
        """Applies tolerance only if value is above the threshold."""
        if value > threshold:
            return (value - offset, value + offset)
        return (value, value)  # No offset applied

    if isinstance(gt, (list, tuple)) and len(gt) == 2:
        start, end = gt
        start_range, end_range = apply_offset(start)
        _, end_range = apply_offset(end)  # Use end's offset if it's also above the threshold
        return start_range <= pred <= end_range
    elif isinstance(gt, (int, float)):
        low, high = apply_offset(gt)
        return low <= pred <= high
    else:
        raise ValueError("Ground truth (gt) must be a number or a list/tuple of two numbers.")


