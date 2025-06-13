import os
import re
import json
import logging
import warnings
import string
from pathlib import Path

import numpy as np
import pandas as pd
import json_repair

from src.utils.localization_utils import (
    load_json,
    clean_prediction_dicts,
    is_missing_prediction,
    normalize_word_level_output,
    calculate_precision_recall_f1_multiple
)

def safe_float_conversion(value):
    """Converts a value to a float, returns None if conversion fails."""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str) and value.isdigit():
        return float(value)
    else:
        return None


def evaluate_word_localization(predictions_filepath, groundtruth_filepath):
    """
    Evaluate word-level and frame-level localization for a given audio dataset.

    Parameters:
        groundtruth_filepath (str): Path to the groundtruth JSON file.
            Expected format:
                [
                    {
                        "audio": "example_audio.opus",
                        "groundtruth": [
                            {"word": "example", "start": 1.2, "end": 2.5},
                            ...
                        ],
                        "question": "Your task is to ..."
                    },
                    ...
                ]


        predictions_filepath (str): Path to the model predictions JSON file.
         Expected format:
                [
                    {
                        "audio": "example_audio.opus",
                        "prediction": (str) JSON string list of dicts with "word", "start", "end" keys.
                            Example: ```json\n[\n {'word': "It's", 'start': 0.0, 'end': 0.16}, ...]```
                        "prompt": "Your task is to ..."
                    },
                    ...
                ]
    Returns:
       metrics(dict): Dictionary :
            - Precision (float)
            - Recall (float)
            - F1-score (float)

    """
    # Load predictions
    model_predictions = pd.DataFrame(load_json(predictions_filepath))
    model_predictions["prediction"] = model_predictions["prediction"].apply(
        lambda x: "" if is_missing_prediction(x) else x
    )
    missing_count = (model_predictions["prediction"] == "").sum()
    if missing_count:
        logging.warning(f"{missing_count} missing predictions will be scored as zero.")

    # Parse predictions
    model_predictions["post_processed"] = model_predictions["prediction"].apply(json_repair.loads)

    # remove empty dictionaries
    model_predictions["post_processed"] = model_predictions["prediction"].apply(json_repair.loads)
    model_predictions["post_processed"] = model_predictions["post_processed"].apply( lambda x: clean_prediction_dicts(x))
    model_predictions["audio"] = model_predictions["audio"].apply(os.path.basename)

    # Load groundtruth
    groundtruth_data =  pd.DataFrame(load_json(groundtruth_filepath))

    # Merge predictions with groundtruth
    merged = groundtruth_data.merge(model_predictions, on="audio")

    # Normalize words; remove punctuations and convert to lowercase
    merged["normalized_predictions"] = merged["post_processed"].apply(
        lambda x: normalize_word_level_output(x, "word")
    )
    merged["normalized_groundtruth"] = merged["groundtruth"].apply(
        lambda x: normalize_word_level_output(x, "word")
    )


    merged['predictions_tuple'] = merged['normalized_predictions'].apply(
    lambda x: [
        (
            d.get("word"),
            safe_float_conversion(d.get("start")),
            safe_float_conversion(d.get("end")),
        )
        if isinstance(d, dict)
        else (None, None, None)  # Return None tuple for non-dict elements
        for d in (x if isinstance(x, list) else [])
    ]
)

    merged["ground_truth_tuple"] = merged['normalized_groundtruth'].apply(
    lambda x: [
        (
            d.get("word"),
            safe_float_conversion(d.get("start")),
            safe_float_conversion(d.get("end")),
        )
        if isinstance(d, dict)
        else (None, None, None)  # Return None tuple for non-dict elements
        for d in (x if isinstance(x, list) else [])
    ]
)
    # Precision, Recall, F1
    precision_overlap, recall_overlap, f1_overlap = calculate_precision_recall_f1_multiple(
         list(merged["predictions_tuple"]), list(merged["ground_truth_tuple"]),
        criterion_kwargs={'threshold': 0.2}  # Using 0.2 seconds threshold
    )

    print(f"Precision: {precision_overlap:.3f}")
    print(f"Recall: {recall_overlap:.3f}")
    print(f"F1-score: {f1_overlap:.3f}")

    metrics = {"Precision": precision_overlap, "Recall": recall_overlap, "f1_score": f1_overlap}

    return metrics


