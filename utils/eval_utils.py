import json
import re
import string
import numpy as np
import json_repair


def load_json(json_file):
    """Load a JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data has been saved to {file_path}")


def read_file_into_list(file_path):
    """Read a file line by line into a list."""
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


def is_missing_prediction(pred):
    """Check if prediction is missing or unparsable."""
    if not pred or pred == "None":
        return True
    try:
        repaired = json_repair.loads(pred.replace("\\", ""))
        return not (isinstance(repaired, list) and all(isinstance(d, dict) for d in repaired))
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
# Word-level utils

def normalize_word_level_output(list_of_dictionaries, key):
    """
    Processes a list of dictionaries by filtering out invalid entries and normalizing text data.

    - Ensures the input is a list, returning None otherwise.
    - Filters out empty dictionaries and those missing the required keys: "word", "start", and "end".
    - Normalizes the values associated with the given key by removing punctuation from its string values.
    - Returns a list of dictionaries with the normalized values.
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




def needleman_wunsch_algorithm(seq1, seq2, match_score=1, mismatch_penalty=-3, gap_penalty=-1):
    """
    seq1: Groundtruth
    seq2: Predictions
    If seq2 is None, return seq1 as is and None.
    """

    if seq2 is None:
        return seq1, None  # Directly return seq1 and None

    words1 = [item['word'] for item in seq1]
    words2 = [item['word'] for item in seq2]

    n = len(words1) + 1
    m = len(words2) + 1
    dp = np.zeros((n, m))
    traceback = np.zeros((n, m), dtype='object')

    # Initialize the DP table and traceback pointers
    for i in range(1, n):
        dp[i][0] = dp[i-1][0] + gap_penalty
        traceback[i][0] = 'U'  # Up for gap in seq2
    for j in range(1, m):
        dp[0][j] = dp[0][j-1] + gap_penalty
        traceback[0][j] = 'L'  # Left for gap in seq1

    # Fill the DP table
    for i in range(1, n):
        for j in range(1, m):
            score = match_score if words1[i-1] == words2[j-1] else mismatch_penalty
            diag = dp[i-1][j-1] + score
            up = dp[i-1][j] + gap_penalty
            left = dp[i][j-1] + gap_penalty
            dp[i][j] = max(diag, up, left)

            # Traceback pointers
            if dp[i][j] == diag:
                traceback[i][j] = 'D'  # Diagonal
            elif dp[i][j] == up:
                traceback[i][j] = 'U'  # Up
            else:
                traceback[i][j] = 'L'  # Left

    # Traceback to align sequences
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = len(words1), len(words2)
    gap_token = {'word': None, 'start': 0, 'end': 0}

    while i > 0 or j > 0:
        if traceback[i][j] == 'D':
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif traceback[i][j] == 'U':
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(gap_token)  # Gap in seq2
            i -= 1
        elif traceback[i][j] == 'L':
            aligned_seq1.append(gap_token)  # Gap in seq1
            aligned_seq2.append(seq2[j-1])
            j -= 1

    # Reverse the aligned sequences
    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]

    # Calculate mismatched words percentage
    mismatched_indices = []
    total_comparisons = 0

    for idx, (item1, item2) in enumerate(zip(aligned_seq1, aligned_seq2)):
        word1 = item1['word'] if item1 else None
        word2 = item2['word'] if item2 else None
        if word1 != word2:
            mismatched_indices.append(idx)
        if word1 or word2:
            total_comparisons += 1

    mismatched_percentage = (len(mismatched_indices) / total_comparisons) * 100 if total_comparisons > 0 else 0

    # Print results
    #print(f"Percentage of misaligned words: {mismatched_percentage:.2f}%")
    #print(f"Mismatched indices: {mismatched_indices}")

    return aligned_seq1, aligned_seq2



def calculate_overall_accuracy(ground_truths, predictions, threshold=0.2):
    """
    Calculate overall accuracy for multiple instances of word timings.

    Parameters:
    ground_truths (list of lists): List of ground truth lists, each containing word timing dictionaries.
    predictions (list of lists): List of predicted word timing lists, each containing word timing dictionaries.
    threshold (float): Time difference threshold (in seconds) to consider a prediction accurate.

    If predictions are None for any instance, count all ground truth words as incorrect predictions

    Returns:
    float: Overall accuracy as a percentage of accurate predictions over total predictions.
    """
    total_accurate_predictions = 0
    total_comparisons = 0

    # Iterate over each pair of ground truth and predicted lists
    for ground_truth, predicted in zip(ground_truths, predictions):
        # If predictions are None, count all ground truth words as incorrect predictions
        if predicted is None:
            total_comparisons += len(ground_truth)  # All GT words are mismatches
            continue

        for gt, pred in zip(ground_truth, predicted):
            # If a prediction is None, it is an incorrect prediction
            if pred is None:
                total_comparisons += 1  # Count the missing prediction
                continue

            # Check if 'start' and 'end' keys are present in both ground truth and predicted data
            if 'start' not in pred or 'end' not in pred or 'start' not in gt or 'end' not in gt:
                total_comparisons += 1  # Count as an incorrect prediction
                continue  # Skip this word

            # Ensure that 'start' and 'end' are numbers (float) for comparison
            try:
                def extract_time(value):
                    """Extracts the numerical time value, handling nested dictionaries."""
                    if isinstance(value, dict):
                        return float(value.get('value', 0))  # Adjust 'value' key if necessary
                    return float(value)

                gt_start = extract_time(gt['start'])
                gt_end = extract_time(gt['end'])
                pred_start = extract_time(pred['start'])
                pred_end = extract_time(pred['end'])

            except (ValueError, TypeError):
                total_comparisons += 1  # Count as an incorrect prediction
                continue  # Skip this word

            # Check if both start and end times are within the threshold
            if abs(pred_start - gt_start) <= threshold and abs(pred_end - gt_end) <= threshold:
                total_accurate_predictions += 1

            # Increment total comparisons for valid entries
            total_comparisons += 1

    # Avoid division by zero
    if total_comparisons == 0:
        return {"error": "No valid predictions available for accuracy calculation"}

    # Calculate overall accuracy
    overall_accuracy = (total_accurate_predictions / total_comparisons) * 100
    return overall_accuracy



def calculate_word_timing_metrics_multiple_instances(ground_truths, predictions):
    """
    https://www.isca-archive.org/interspeech_2020/sainath20_interspeech.pdf

    Calculate word timing metrics over multiple instances of ground truth and predicted data.
    If predictions is None or contains None for specific instances, those cases are factored into the calculation.

    Parameters:
    ground_truths (list of lists): List of ground truth lists, each containing word timing dictionaries.
    predictions (list of lists or None): List of predicted word timing lists, each containing word timing dictionaries or None.

    Returns:
    dict: Dictionary with the aggregated metrics:
          {'Ave. ST ∆': value, 'Ave. ET ∆': value, '% WS < 200ms': value, '% WE < 200ms': value}
    """
    total_start_time_deltas = []
    total_end_time_deltas = []
    total_ws_less_200ms_count = 0
    total_we_less_200ms_count = 0
    total_words = 0

    def extract_time(value):
        """Extracts the numerical time value, handling nested dictionaries and lists."""
        if isinstance(value, dict):
            return float(value.get('value', 0))  # Adjust if necessary
        elif isinstance(value, list) and value:  # If it's a list, take the first item
            return extract_time(value[0])
        return float(value)

    for ground_truth, predicted in zip(ground_truths, predictions):
        # If the prediction for this instance is None, treat all words as mismatched
        if predicted is None:
            predicted = [{'start': float('inf'), 'end': float('inf')}] * len(ground_truth)

        # Ensure that the number of predictions matches the ground truth
        instance_words = len(ground_truth)
        total_words += instance_words

        for gt, pred in zip(ground_truth, predicted):
            # Ensure 'start' and 'end' are present; if pred was None, it was replaced with inf above
            try:
                gt_start = extract_time(gt['start'])
                gt_end = extract_time(gt['end'])
                pred_start = extract_time(pred['start'])
                pred_end = extract_time(pred['end'])
            except (ValueError, TypeError) as e:
                print(f"Skipping due to error: {e} | Data: gt={gt}, pred={pred}")
                continue

            # If pred_start or pred_end were inf, it means there was no valid prediction
            if pred_start == float('inf') or pred_end == float('inf'):
                delta_start = float('inf')
                delta_end = float('inf')
            else:
                delta_start = abs(pred_start - gt_start)
                delta_end = abs(pred_end - gt_end)

            total_start_time_deltas.append(delta_start)
            total_end_time_deltas.append(delta_end)

            if delta_start < 0.2:
                total_ws_less_200ms_count += 1
            if delta_end < 0.2:
                total_we_less_200ms_count += 1

    # If no valid words, return an error
    if total_words == 0:
        return {"error": "No valid predictions available for comparison across all instances"}

    # Compute metrics, treating `inf` values as missing data points
    valid_start_deltas = [d for d in total_start_time_deltas if d != float('inf')]
    valid_end_deltas = [d for d in total_end_time_deltas if d != float('inf')]

    ave_st_delta = np.mean(valid_start_deltas) if valid_start_deltas else float('inf')
    ave_et_delta = np.mean(valid_end_deltas) if valid_end_deltas else float('inf')

    ws_less_200ms = (total_ws_less_200ms_count / total_words) * 100
    we_less_200ms = (total_we_less_200ms_count / total_words) * 100

    return {
        'Ave. ST ∆': ave_st_delta,
        'Ave. ET ∆': ave_et_delta,
        '% WS < 200ms': ws_less_200ms,
        '% WE < 200ms': we_less_200ms
    }

def calculate_precision_recall_f1(ground_truths, predictions, threshold=0.2):
    """
    Calculate precision, recall, and F1-score for word timing predictions.

    Parameters:
    - ground_truths (list of lists): List of ground truth lists, each containing word timing dictionaries.
    - predictions (list of lists): List of predicted word timing lists, each containing word timing dictionaries.
    - threshold (float): Time difference threshold (in seconds) to consider a prediction correct.

    Returns:
    - dict: Precision, recall, and F1-score values.
    """

    def extract_time(value):
        """Extracts the numerical time value, handling nested dictionaries."""
        if isinstance(value, dict):
            return float(value.get('value', 0))  # Adjust 'value' key if necessary
        return float(value)

    true_positives = 0
    false_negatives = 0  # Ground truth words that were not matched correctly
    false_positives = 0  # Predicted words that do not match any ground truth

    for ground_truth, predicted in zip(ground_truths, predictions):
        gt_matched = set()  # Track matched ground truth indices
        pred_matched = set()  # Track matched prediction indices

        if predicted is None:
            # If predictions are missing, count only valid ground truth words as false negatives
            false_negatives += sum(1 for gt in ground_truth if gt.get("word") is not None)
            continue

        for gt_idx, gt in enumerate(ground_truth):
            if gt.get("word") is None or 'start' not in gt or 'end' not in gt:
                continue  # Skip invalid ground truth entries

            gt_start, gt_end = extract_time(gt['start']), extract_time(gt['end'])
            matched = False

            for pred_idx, pred in enumerate(predicted):
                if pred is None or 'start' not in pred or 'end' not in pred:
                    continue  # Skip invalid predictions

                pred_start, pred_end = extract_time(pred['start']), extract_time(pred['end'])

                # Check if the prediction matches the ground truth within the threshold
                if abs(pred_start - gt_start) <= threshold and abs(pred_end - gt_end) <= threshold:
                    true_positives += 1
                    gt_matched.add(gt_idx)
                    pred_matched.add(pred_idx)
                    matched = True
                    break  # Move to the next ground truth word

            if not matched:
                false_negatives += 1  # No matching prediction found for this ground truth word

        # Count false positives: unmatched predictions
        false_positives += len(predicted) - len(pred_matched)

    # Compute precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Compute F1-score (harmonic mean of precision and recall)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}
