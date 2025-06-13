import json
import re
import string
import numpy as np
import json_repair
from typing import List, Tuple, Callable

def load_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data has been saved to {file_path}")

def read_file_into_list(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]

def convert(sec, resolution=0.01):
    """Convert seconds to frame index using specified resolution (default: 10ms)."""
    return int(sec / resolution)


def update_dur_counts(counts, preds, gts):
    """Update true positive, false positive, and false negative counts."""
    for p, g in zip(preds, gts):
        if p == g == 1:
            counts["true_pos"] += 1
        elif p == 1:
            counts["false_pos"] += 1
        elif g == 1:
            counts["false_neg"] += 1


def update_false_neg_cnt(counts, missing_indices, gt_dict, eval_type="segment"):
    """Update false negative counts for missing predictions."""
    for idx in missing_indices:
        entries = gt_dict.get(idx, [])
        if eval_type == "segment":
            counts["false_neg"] += len(entries)
        else:
            for phrase, start, end in entries:
                if eval_type == "word":
                    counts["false_neg"] += len(phrase.split())
                elif eval_type == "frame":
                    counts["false_neg"] += convert(end) - convert(start)


def undetected_indices(gt_dict, pred_dict):
    """Return list of indices present in ground truth but not in prediction."""
    return list(set(gt_dict) - set(pred_dict))


def convert_to_list_of_dicts(item):
    """Ensure input is a list of dictionaries."""
    if isinstance(item, dict):
        return [item]
    if isinstance(item, list) and all(isinstance(x, dict) for x in item):
        return item
    return None


def is_missing_span_prediction(pred):
    """Check if prediction is missing or unparsable."""
    if not pred or pred == "None":
        return True
    try:
        repaired = json_repair.loads(pred.replace("\\", ""))
        return not (isinstance(repaired, list) and all(isinstance(d, dict) for d in repaired))
    except Exception:
        return True


def clean_prediction_dicts(pred_list):
    """
    Remove dicts that are empty or missing 'word', 'start', or 'end' with valid values.
    """
    required_keys = {'word', 'start', 'end'}
    return [
        d for d in pred_list
        if isinstance(d, dict)
        and required_keys.issubset(d)
        and all(d[k] not in [None, ""] for k in required_keys)
    ]

def is_missing_prediction(pred):
    """Check if prediction is missing or unparsable."""
    if not pred or pred == "None":
        return True
    try:
        repaired = json_repair.loads(pred)
        #print(len(repaired))

        # Ensure it's a list and filter out empty dictionaries
        if isinstance(repaired, list):
            repaired = [d for d in repaired if isinstance(d, dict) and d]  # keep only non-empty dicts
            repaired = clean_prediction_dicts(repaired)

            return not repaired  # return True if the list is now empty
        else:
            return True
    except Exception:
        return True


def remove_invalid_rows(lst):
    """Remove rows with None or empty string elements."""
    if not lst:
        return None
    return [
        row for row in lst
        if isinstance(row, (list, tuple)) and all(el not in [None, ""] for el in row)
    ]


def remove_invalid_lists(lst):
    """Keep only 3-element lists."""
    return [x for x in lst if len(x) == 3]


def extract_values(lst):
    """Extract values from list of dictionaries."""
    return [list(d.values()) for d in lst]


def normalize_span_output(data, key):
    """Normalize text spans by removing punctuation and converting to lowercase."""
    if not isinstance(data, list) or any(not item for item in data):
        return None
    return [
        {**item, key: item[key].translate(str.maketrans("", "", string.punctuation))#.lower()
        }
        for item in data if key in item and isinstance(item[key], str)
    ]

def convert_time(value):
    """
    Convert various time formats to seconds.
    Supported formats:
    - float / int
    - "ss:SSS", "mm:ss.SSS", "mm:ss:SSS", "hh:mm:ss.SSS", "mm.ss.SSS"
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise TypeError(f"Unsupported type: {type(value)}")

    try:
        return float(value)
    except ValueError:
        pass

    patterns = [
        (r"(\d+):(\d+)", lambda m: int(m[0]) + int(m[1]) / 1000),
        (r"(\d+):(\d+)\.(\d+)", lambda m: int(m[0]) * 60 + int(m[1]) + int(m[2]) / 1000),
        (r"(\d+):(\d+):(\d+)", lambda m: int(m[0]) * 60 + int(m[1]) + int(m[2]) / 1000),
        (r"(\d+):(\d+):(\d+)\.(\d+)", lambda m: int(m[0]) * 3600 + int(m[1]) * 60 + int(m[2]) + int(m[3]) / 1000),
        (r"(\d+)\.(\d+)\.(\d+)", lambda m: int(m[0]) * 60 + int(m[1]) + int(m[2]) / 1000),
    ]

    for pattern, parser in patterns:
        match = re.fullmatch(pattern, value)
        if match:
            return parser(match.groups())

    raise ValueError(f"Unrecognized time format: {value}")


def extract_values_by_keys(lst, keys):
    """
    Extract specified keys from a list of dictionaries and convert time values.
    Filters out entries with invalid or missing time values.
    """
    def is_valid(val):
        if val in {None, ""}:
            return False
        if isinstance(val, (int, float)):
            return True
        return val.replace(":", "").replace(".", "").isdigit()

    def process(d):
        return [convert_time(d[k]) if k in {keys[1], keys[2]} else d[k] for k in keys]

    if not isinstance(lst, list):
        return lst

    filtered = [d for d in lst if is_valid(d.get(keys[1])) and is_valid(d.get(keys[2]))]
    return [process(d) for d in filtered]


#--------------------
# Word-level  utils

def normalize_word_level_output(list_of_dictionaries, key):
    """
    Processes a list of word localization predictions dictionaries by filtering out invalid entries and normalizing words.

    - Ensures the input is a list, returning None otherwise.
    - Filters out empty dictionaries and those missing the required keys: "word", "start", and "end".
    - Normalizes the values associated with the given key by removing punctuation from its string values.

    """
    # Check if the input is a list
    if not isinstance(list_of_dictionaries, list):
        return None

    # Remove dictionaries that don't have the required keys or are empty
    list_of_dictionaries = [
        item for item in list_of_dictionaries
        if isinstance(item, dict)  # Ensure item is a dictionary
        and all(k in item for k in ["word", "start", "end"])  # Ensure required keys are present
        and item  # Remove empty dictionaries
    ]

    # Normalize output by removing punctuation from the specified key's value
    normalized_output = [
        {**item, key: item[key].translate(str.maketrans('', '', string.punctuation))}
        for item in list_of_dictionaries
        if key in item and isinstance(item[key], str)  # Ensures key exists and value is a string
    ]

    return normalized_output


def overlap_criterion(
    prediction_interval: Tuple[float, float],
    ground_truth_interval: Tuple[float, float],
    threshold: float = 0.2  # Threshold here is 200 ms
) -> bool:
    """
    Checks if two time intervals are within a threshold of each other.
    """
    if any(v is None for v in prediction_interval + ground_truth_interval):
        return False
    start_pred, end_pred = prediction_interval
    start_gt, end_gt = ground_truth_interval
    return abs(start_pred - start_gt) <= threshold and abs(end_pred - end_gt) <= threshold


def calculate_precision_recall_f1_multiple(
    predictions: List[List[Tuple[str, float, float]]],
    ground_truth: List[List[Tuple[str, float, float]]],
    matching_criterion: Callable[
        [Tuple[float, float], Tuple[float, float]], bool
    ] = overlap_criterion,  # Default to overlap criterion
    criterion_kwargs: dict = None  # kwargs for matching criterion
) -> Tuple[float, float, float]:
    """
    Evaluates word timestamp predictions against ground truth for multiple audio files
    using a matching criterion (200ms ) . Handles cases where predictions for a file are None.

    Parameters:
        predictions: A list of lists of predicted word timestamps, where each inner list
            corresponds to one audio file: [[(word, start, end), ...], ...].
        ground_truth: A list of lists of ground truth word timestamps, with the same
            structure as predictions.
        matching_criterion: A function that determines if a predicted interval
            matches a ground truth interval. Defaults to overlap_criterion.
        criterion_kwargs: Keyword arguments to pass to the
            matching_criterion function.

    """
    if len(predictions) != len(ground_truth):
        raise ValueError("The number of prediction lists must match the number of ground truth lists.")

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    if criterion_kwargs is None:
        criterion_kwargs = {}

    for pred_list, gt_list in zip(predictions, ground_truth):
        if pred_list is None:
            if gt_list:  # If there are ground truth items and no predictions
                total_false_negatives += len(gt_list)
            continue  # Move to the next file

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Create a copy to keep track of which ground truth items have been matched in the current file
        ground_truth_unmatched = list(gt_list)

        for pred_word, pred_start, pred_end in pred_list:
            best_match = None
            best_match_index = -1

            for i, (gt_word, gt_start, gt_end) in enumerate(ground_truth_unmatched):
                if pred_word == gt_word and matching_criterion(
                    (pred_start, pred_end), (gt_start, gt_end), **criterion_kwargs
                ):
                    best_match = (gt_word, gt_start, gt_end)
                    best_match_index = i
                    break  # Stop after finding the first match

            if best_match:
                true_positives += 1
                if best_match_index != -1:
                    del ground_truth_unmatched[best_match_index]  # Remove the matched ground truth
            else:
                false_positives += 1

        false_negatives = len(ground_truth_unmatched)  # Any remaining unmatched ground truth are FNs

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

    if total_true_positives + total_false_positives == 0:
        precision = 0.0
    else:
        precision = total_true_positives / (total_true_positives + total_false_positives)

    if total_true_positives + total_false_negatives == 0:
        recall = 0.0
    else:
        recall = total_true_positives / (total_true_positives + total_false_negatives)

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision * 100, recall * 100, f1_score * 100



