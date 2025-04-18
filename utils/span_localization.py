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

# Suppress warnings
#warnings.filterwarnings("ignore")

# Custom utilities
from eval_utils import (
    convert,
    convert_time,
    convert_to_list_of_dicts,
    extract_values,
    extract_values_by_keys,
    is_missing_prediction,
    load_json,
    normalize_span_output,
    remove_invalid_lists,
    remove_invalid_rows,
    save_json,
    undetected_indices,
    update_dur_counts,
    update_false_neg_cnt,
)


def process_gt_word(cnt_dct, pred_tuple_lst, gt_tuple, pred_idx, thresh):
    wrd, gt_start, gt_end = gt_tuple
    gt_len = gt_end - gt_start
    is_sil = wrd == "" or wrd == "#"  # silence
    is_entity = len(wrd) > 1 and wrd[0] == "#"  # non-silence and word in entity phrase
    #print(f"pred_tuple_lst is {pred_tuple_lst[pred_idx]}")
    _, pred_start, pred_end = pred_tuple_lst[pred_idx]

    pred_start, pred_end = convert_time(pred_start), convert_time(pred_end)

    if is_sil:
        if not pred_end > gt_end:
            return pred_idx + 1
        else:
            return pred_idx

    #print(f"pred_start is {pred_start} and gt_end is {gt_end}")
    #print(f"type pred_start is {type(pred_start)} and type gt_end is {type(gt_end)}")

    if not pred_start < gt_end:
        if is_entity:
            cnt_dct["false_neg"] += 1
        return pred_idx  # current pred tuple not processed


    tot_overlap_dur = np.min([pred_end, gt_end]) - np.max([pred_start, gt_start])

    while not pred_end > gt_end:
        if len(pred_tuple_lst) > pred_idx + 1:
            pred_idx += 1
            _, pred_start, pred_end = pred_tuple_lst[pred_idx]

            pred_start, pred_end = convert_time(pred_start), convert_time(pred_end)
            if pred_start < gt_end:
                tot_overlap_dur += np.min([pred_end, gt_end]) - np.max(
                    [pred_start, gt_start]
                )
        else:
            pred_idx += 1
            break

    overlap_ratio = tot_overlap_dur / gt_len
    if is_entity and overlap_ratio < thresh:  # but thresh not met
        cnt_dct["false_neg"] += 1
    elif is_entity:
        cnt_dct["true_pos"] += 1
    elif not overlap_ratio < thresh:  # but not an entity
        cnt_dct["false_pos"] += 1

    return pred_idx



def convert_time_to_frame_idx(pred_lst, gt_lst):
    """
    convert time stamps to frame index
    """

    def create_array(num_frames, tuple_lst, arr_type):
        arr = np.zeros(num_frames)
        if arr_type == "pred":
            for _, start, end in tuple_lst:
                arr[convert(start) : convert(end)] = 1
        else:
            # Initialize the previous_end_time variable to track the end time of the previous entity segment
            previous_end_time = None
            is_in_entity_span = False
            start_time = None

            for seg, start, end in tuple_lst:
                if len(seg) > 0 and seg[0] == "#":  # Check if the segment is an entity
                    if not is_in_entity_span:
                        # Start of a new entity span, capture the start time
                        start_time = start
                        is_in_entity_span = True  # Mark as inside an entity span

                    # Update previous_end_time to the current segment's end
                    previous_end_time = end

                else:  # Encounter a non-entity segment
                    if is_in_entity_span:
                        # We were in an entity span and now encounter a non-entity, so we mark the span
                        end_time = previous_end_time  # Use the previous segment's end time
                        arr[convert(start_time):convert(end_time)] = 1  # Mark the span in the array
                        is_in_entity_span = False  # Reset the flag as we're no longer inside an entity span

            # If the list ends and we're still inside an entity span, finalize it
            if is_in_entity_span:
                end_time = previous_end_time  # Use the last segment's end time
                arr[convert(start_time):convert(end_time)] = 1  # Mark the span


        return arr

    tot_time = np.max([float(pred_lst[-1][1]), float(gt_lst[-1][2])])
    num_frames = convert(tot_time)

    pred_array = create_array(num_frames, pred_lst, "pred")
    gt_array = create_array(num_frames, gt_lst, "gt")
    return pred_array, gt_array


def evaluate(true_pos, false_neg, false_pos):
    if true_pos == 0:
        return 0, 0, 0
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_alignments_word(gt_alignment_dct, pred_tuple_dct, gt_tuple_dct, thresh=1):
    """
    Word-level evaluation for alignment:

    - True Positive (TP): Correctly aligned entity words.
    - False Negative (FN): Ground truth entity words not matched by prediction.
    - False Positive (FP): Non-entity words predicted as entities.

    A word match is determined by a minimum overlap threshold (fractional).
    """

    # Initial false negative detection (e.g., utterances with no predictions)

    false_neg_indices = []
    false_neg_indices.extend(undetected_indices(gt_tuple_dct, pred_tuple_dct))

    cnt_dct = {"true_pos": 0, "false_neg": 0, "false_pos": 0}

    for utt_idx, gt_tuple_lst in gt_alignment_dct.items():
        preds = pred_tuple_dct.get(utt_idx)

        if isinstance(preds, list) and preds:  # Non-empty list of predictions
            pred_idx = 0
            for gt_tuple in gt_tuple_lst:
                if pred_idx < len(preds):
                    pred_idx = process_gt_word(cnt_dct, preds, gt_tuple, pred_idx, thresh)
        else:
            # Treat None or empty list as a false negative
            if utt_idx in gt_tuple_dct and gt_tuple_dct[utt_idx]:
                false_neg_indices.append(utt_idx)

    update_false_neg_cnt(cnt_dct, false_neg_indices, gt_tuple_dct, "word")

    return evaluate(cnt_dct["true_pos"], cnt_dct["false_neg"], cnt_dct["false_pos"])




def evaluate_alignments_frames(gt_alignment_dct, pred_tuple_dct, gt_tuple_dct):
    """
    Frame-level measure:
    Each frame is evaluated as a hit (TP) or a miss (FN).
    FP: # non-entity frames redacted
    """
    false_neg_indices = []
    false_neg_indices.extend(undetected_indices(gt_tuple_dct, pred_tuple_dct))

    cnt_dct = {"true_pos": 0, "false_neg": 0, "false_pos": 0}

    for idx, pred_tuple_lst in pred_tuple_dct.items():
        if isinstance(pred_tuple_lst, list) and pred_tuple_lst:
            if idx in gt_alignment_dct:
                pred_array, gt_array = convert_time_to_frame_idx(
                    pred_tuple_lst, gt_alignment_dct[idx]
                )
                update_dur_counts(cnt_dct, pred_array, gt_array)
            else:
                for start_time, end_time in pred_tuple_lst:
                    cnt_dct["false_pos"] += convert(end_time) - convert(start_time)
        elif idx in gt_tuple_dct and gt_tuple_dct[idx]:
            # Treat None or empty list as a false negative
            false_neg_indices.append(idx)

    update_false_neg_cnt(cnt_dct, false_neg_indices, gt_tuple_dct, "frame")
    return evaluate(cnt_dct["true_pos"], cnt_dct["false_neg"], cnt_dct["false_pos"])




def evaluate_submission(gt_alignment_dct, pred_dct, gt_dct, offset=-1, ms=False):
    res_dct = {
        "word": {"f1": {}, "prec": {}, "recall": {}},
        "frame": {},
    }
    frac_lst = [1, 0.9, 0.8, 0.7, 0.6, 0.5]

    for frac_tol in frac_lst:
        prec, recall, f1 = evaluate_alignments_word(
            gt_alignment_dct, pred_dct, gt_dct, frac_tol
        )
        res_dct["word"]["f1"][frac_tol] = f1
        res_dct["word"]["prec"][frac_tol] = prec
        res_dct["word"]["recall"][frac_tol] = recall

    #print(f"pred_dct is {pred_dct}")
    prec, recall, f1 = evaluate_alignments_frames(gt_alignment_dct, pred_dct, gt_dct)

    res_dct["frame"]["f1"] = f1
    res_dct["frame"]["prec"] = prec
    res_dct["frame"]["recall"] = recall

    print("Frame-F1: ", np.round(100 * res_dct["frame"]["f1"], 2))

    return res_dct



def evaluate_span_localization(groundtruth_filepath, predictions_filepath, task):
    """
    Evaluate span-level and frame-level localization for a given task.

    Parameters:
        groundtruth_filepath (str): Path to the groundtruth JSON file.
            Expected format:
            [
                {
                    "audio": "ads_local_1.opus",
                    "groundtruth": {
                        "entities" or "ads_segment": [
                            {"advertisement" or "entity": "...", "start": float, "end": float},
                            ...
                        ],
                        "word_timestamps": [
                            {"word": "...", "start": float, "end": float},
                            ...
                        ]
                    },
                    "question": "Your task is to analyze ..."
                },
                ...
            ]

        predictions_filepath (str): Path to the model predictions JSONL file.
            Each line should be a JSON object with keys:
            - "audio": audio file name
            - "prompt": same prompt as in the groundtruth
            - "prediction": string list of dictionaries, e.g.
              "[{\"advertisement\": \"Buy now...\", \"start\": 13.2, \"end\": 44.8}, ...]"

        task (str): Task name, either "advertisement" or "named_entity".

    Returns:
        results_df (pd.DataFrame): Merged DataFrame containing:
            - Groundtruth and prediction spans
            - Preprocessed word-level timestamps
            - Instance-level frame F1 scores

        final_score (dict): Dictionary with global frame-level precision, recall, and F1.
    """

    def extract_base_audio(df):
        df["audio"] = df["audio"].apply(os.path.basename)
        return df

    def process_predictions(df):
        # Identify and handle missing predictions
        missing_mask = df["prediction"].apply(is_missing_prediction)
        missing_count = missing_mask.sum()
        if missing_count > 0:
            logging.warning(f"{missing_count} missing predictions (None or malformed) will be scored as zero.")
            df.loc[missing_mask, "prediction"] = ""

        df["formatted_predictions"] = df["prediction"].apply(
            lambda x: convert_to_list_of_dicts(json_repair.loads(x.replace("\\", "")))
        )
        entity_key = "advertisement" if task == "advertisement" else "entity"
        df["normalized_predictions"] = df["formatted_predictions"].apply(
            lambda x: normalize_span_output(x, entity_key)
        )
        df["preprocessed_predictions"] = df["normalized_predictions"].apply(
            lambda x: remove_invalid_rows(extract_values_by_keys(x, [entity_key, "start", "end"]))
        )
        return df

    def process_groundtruth(df):
        df = df.rename(columns={"question": "prompt"})
        df = extract_base_audio(df)
        if task == "advertisement":
            df[["word_timestamp", "span"]] = pd.json_normalize(df["groundtruth"])[["word_timestamp", "ads_segment"]]
            gt_keys = ["text", "start", "end"]
        else:
            df[["word_timestamp", "span"]] = pd.json_normalize(df["groundtruth"])[["word_timestamp", "entities"]]
            gt_keys = ["entity", "start", "end"]


        df["normalized_span"] = df["span"].apply(
            lambda x: normalize_span_output(x, gt_keys[0])
        )
        df["preprocessed_groundtruth"] = df["normalized_span"].apply(lambda x: extract_values_by_keys(x, gt_keys))
        df["preprocessed_word_timestamps"] = df["word_timestamp"].apply(
            lambda x: remove_invalid_lists(extract_values(x))
        )
        return df

    # Load and process predictions
    model_predictions = pd.read_json(predictions_filepath, orient="records", lines=True)
    model_predictions = extract_base_audio(model_predictions)
    model_predictions = process_predictions(model_predictions)

    # Load and process groundtruth
    groundtruth_data = pd.read_json(groundtruth_filepath)
    groundtruth_data = process_groundtruth(groundtruth_data)

    # Merge and construct identifiers
    if task == "advertisement":
        merged_df = groundtruth_data.merge(model_predictions, on=["audio"])
    else:
        merged_df = groundtruth_data.merge(model_predictions, on=["audio", "prompt"])

    merged_df["audio_with_index"] = merged_df["audio"] + "_" + merged_df.index.astype(str)

    # Convert to dicts for evaluation
    span_groundtruth_dict = dict(zip(merged_df["audio_with_index"], merged_df["preprocessed_groundtruth"]))
    word_timestamp_dict = dict(zip(merged_df["audio_with_index"], merged_df["preprocessed_word_timestamps"]))
    span_prediction_dict = dict(zip(merged_df["audio_with_index"], merged_df["preprocessed_predictions"]))

    # Evaluate full dataset
    final_score_nel = evaluate_submission(word_timestamp_dict, span_prediction_dict, span_groundtruth_dict)

    # Instance-level F1
    def to_single_audio_dict(row, col):
        return {row["audio"]: row[col]}

    merged_df["groundtruth_dict"] = merged_df.apply(lambda x: to_single_audio_dict(x, "preprocessed_groundtruth"), axis=1)
    merged_df["word_timestamp_dict"] = merged_df.apply(lambda x: to_single_audio_dict(x, "preprocessed_word_timestamps"), axis=1)
    merged_df["prediction_dict"] = merged_df.apply(lambda x: to_single_audio_dict(x, "preprocessed_predictions"), axis=1)

    merged_df["scores"] = merged_df.apply(
        lambda x: evaluate_submission(x.word_timestamp_dict, x.prediction_dict, x.groundtruth_dict),
        axis=1
    )
    merged_df["frame-f1"] = merged_df["scores"].apply(lambda x: x["frame"]["f1"] * 100)

    return merged_df, final_score_nel
