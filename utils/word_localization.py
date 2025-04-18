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

# Custom utilities
from eval_utils import (
    is_missing_prediction,
    normalize_word_level_output,
    needleman_wunsch_algorithm,
    calculate_overall_accuracy,
    calculate_word_timing_metrics_multiple_instances,
    calculate_precision_recall_f1
)



def evaluate_word_localization(groundtruth_filepath, predictions_filepath):
    """
    Evaluate word-level and frame-level localization for a given audio dataset.

    Parameters:
        groundtruth_filepath (str): Path to the groundtruth JSON file.
            Expected format:
                [
                    {
                        "audio": "example_audio.opus",
                        "groundtruth": [
                            {"word": "example", "start": float, "end": float},
                            ...
                        ],
                        "question": "Your task is to ..."
                    },
                    ...
                ]

        predictions_filepath (str): Path to the model predictions JSONL file.
            Each line should be a JSON object with:
                - "audio": (str) audio file name
                - "prompt": (str) same prompt as groundtruth
                - "prediction": (str) JSON string list of dicts with "word", "start", "end" keys
                    Example: '[{"word": "Buy", "start": 13.2, "end": 44.8}, ...]'

    Returns:
        model_predictions (pd.DataFrame): DataFrame containing the cleaned and parsed predictions.
        groundtruth_data (pd.DataFrame): DataFrame containing the groundtruth information.
    """
    # Load predictions
    model_predictions = pd.read_json(predictions_filepath, orient="records", lines=True)
    model_predictions["prediction"] = model_predictions["prediction"].apply(
        lambda x: "" if is_missing_prediction(x) else x
    )
    missing_count = (model_predictions["prediction"] == "").sum()
    if missing_count:
        logging.warning(f"{missing_count} missing predictions will be scored as zero.")

    # Parse predictions
    model_predictions["post_processed"] = model_predictions["prediction"].apply(json_repair.loads)
    model_predictions["audio"] = model_predictions["audio"].apply(os.path.basename)

    # Load groundtruth
    groundtruth_data = pd.read_json(groundtruth_filepath)

    # Merge predictions with groundtruth
    merged = groundtruth_data.merge(model_predictions, on="audio")

    # Normalize words; remove punctuations and convert to lowercase
    merged["normalized_predictions"] = merged["post_processed"].apply(
        lambda x: normalize_word_level_output(x, "word")
    )
    merged["normalized_groundtruth"] = merged["groundtruth"].apply(
        lambda x: normalize_word_level_output(x, "word")
    )

    # Align predictions and groundtruth
    merged["aligned"] = merged.apply(
        lambda row: needleman_wunsch_algorithm(row["normalized_groundtruth"], row["normalized_predictions"]),
        axis=1
    )
    merged["groundtruth_aligned"], merged["prediction_aligned"] = zip(*merged["aligned"])

    # Accuracy
    accuracy = calculate_overall_accuracy(
        list(merged["groundtruth_aligned"]),
        list(merged["prediction_aligned"])
    )
    print(f"Accuracy: {accuracy:.2%}")

    # Timing metrics
    timing_metrics = calculate_word_timing_metrics_multiple_instances(
        list(merged["groundtruth_aligned"]),
        list(merged["prediction_aligned"])
    )
    print(f"Word timing metrics: {timing_metrics}")
    timing_metrics["Accuracy"] = accuracy
    timing_metrics = {f"full_{k}": v for k, v in timing_metrics.items()}

    # Precision, Recall, F1
    prf = calculate_precision_recall_f1(
        list(merged["groundtruth_aligned"]),
        list(merged["prediction_aligned"])
    )
    print(f"Precision, Recall, F1: {prf}")

    return model_predictions, groundtruth_data






