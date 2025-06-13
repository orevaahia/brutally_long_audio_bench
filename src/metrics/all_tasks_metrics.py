import os
import re
import pandas as pd
from typing import Optional

from src.utils.numeric_utils import (
    load_json,
    safe_literal_eval,
    is_correct,
    is_correct_threshold_offset,
    convert_minutes_to_seconds,
    convert_to_seconds,
    extract_integer,
    remove_bracketed_tags
)
from src.utils.span_localization import evaluate_span_localization
from src.utils.word_localization import evaluate_word_localization

class LocalizationMetric:
    def __init__(self, groundtruth_filepath, predictions_filepath, task):
        self.predictions_filepath = predictions_filepath
        self.groundtruth_filepath = groundtruth_filepath
        self.metrics = {}
        if task == "word_localization":
            self.eval_word_localization()
        elif task == "named_entity_localization":
            self.ne_localization()
        elif task == "advertisement_localization":
            self.advertisement_localization()

    def eval_word_localization(self ):
        metrics = evaluate_word_localization(self.predictions_filepath, self.groundtruth_filepath)
        self.metrics.update(metrics)
        return metrics

    def ne_localization(self):
        metrics = evaluate_span_localization(self.predictions_filepath, self.groundtruth_filepath, "entity")
        self.metrics.update(metrics)
        return metrics

    def advertisement_localization(self):
        metrics = evaluate_span_localization(self.predictions_filepath, self.groundtruth_filepath, "advertisement")
        self.metrics.update(metrics)
        return metrics

class NonLocalizationMetric:
    def __init__(self,
    groundtruth_json, predictions_json, task, prompt_suffix: Optional[str] = None, offset: Optional[str] = None):

        self.predictions_dataframe = pd.DataFrame(load_json(predictions_json))
        self.predictions_dataframe["audio"] = self.predictions_dataframe["audio"].apply(lambda x: os.path.basename(x))
        self.groundtruth_dataframe = pd.DataFrame(load_json(groundtruth_json))
        self.groundtruth_dataframe["audio"] = self.groundtruth_dataframe["audio"].apply(lambda x: os.path.basename(x))
        self.metrics = {}
        self.offset = offset
        self.prompt_suffix = prompt_suffix
        self.task = task
        if task == "speaker_number_estimation":
            self.eval_speaker_number_estimation()
        elif task == "entire_duration":
            self.eval_entire_duration(offset)
        elif task == "event_duration":
            self.eval_event_duration(offset)
        elif task == "event_duration_short_audio":
            self.eval_event_duration_short_audio(offset)
        elif task == "emotion_ranking":
            self.eval_emotion_tasks(prompt_suffix, "emotion_ranking")
        elif task == "emotion_reasoning":
            self.eval_emotion_tasks(prompt_suffix, "emotion_reasoning")


    @staticmethod
    def create_duration_prompt(prompt, audio_length):
        if audio_length == "long":
            return f"{prompt}. Provide only the numeric value as an integer without any explanation. Do not use the MM:SS format."
        else:
            return f"{prompt}. Return only a numeric integer value without any explanation."

    @staticmethod
    def remove_after_sentence(text, sentence_pattern):
        """
        Removes the first occurrence of a sentence matching a regular expression pattern and everything after it.
        """
        match = re.search(sentence_pattern, text)
        if match:
            return text[:match.start()].strip()
        else:
            return text.strip()

    def _merge_on_audio(self):
        return self.groundtruth_dataframe[
            self.groundtruth_dataframe["audio"].isin(self.predictions_dataframe["audio"])
        ].merge(self.predictions_dataframe, on="audio", how="left")

    def _compute_accuracy(self, df, pred_col, gt_col, offset=None):
        if "emotion" not in self.task:
            df[pred_col] = df[pred_col].apply(lambda row: extract_integer(row))

        df["correct"] = df.apply(lambda row: is_correct(row[pred_col], row[gt_col]), axis=1)
        acc = df["correct"].mean()

        self.metrics["accuracy"] = acc * 100

        if offset is not None:
            df["correct_offset"] = df.apply(
                lambda row: is_correct_threshold_offset(row[pred_col], row[gt_col], offset), axis=1
            )
            acc_offset = df["correct_offset"].mean()
            self.metrics[f"accuracy_with_offset_{offset}_seconds"] = acc_offset * 100

    def eval_speaker_number_estimation(self):
        df = self._merge_on_audio()
        df["groundtruth"] = df["groundtruth"].apply(lambda x: safe_literal_eval(x))
        self._compute_accuracy(df, pred_col="prediction", gt_col="groundtruth")

        return df


    def eval_entire_duration(self, offset=None):
        offset = offset or self.offset
        df = self._merge_on_audio()
        self._compute_accuracy(df, pred_col="prediction", gt_col="groundtruth", offset=offset)

    def eval_emotion_tasks(self, prompt_suffix, task=None):
        df = self.groundtruth_dataframe.copy()
        df["prompt"] = df["question"].apply(lambda x: x.strip())
        self.predictions_dataframe["prompt"] = self.predictions_dataframe["prompt"].apply(lambda x: self.remove_after_sentence(x, prompt_suffix))

        def extract_final_choice(text, options):
            """
            Extract the final predicted option from detailed model output with reasoning.

            Args:
                text (str): The output text from the model.
                options (tuple): The valid options (default A–E).

            Returns:
                str or None: The selected option (e.g., 'B') or None if not found.
            """
            # Look for patterns like **(B)**, (B), or just B near the end
            pattern = r'\*\*\(([A-E])\)\*\*|\(([A-E])\)'  # Matches **(B)** or (B)
            #print(f"text is {text}")
            matches = re.findall(pattern, str(text))
            matches = [m[0] or m[1] for m in matches if m[0] or m[1]]

            # Return the last mentioned valid option — usually the final answer
            for choice in reversed(matches):
                if choice in options:
                    return choice
            return None


        if task == "emotion_ranking":
            merged_df = df.merge(self.predictions_dataframe, on=["audio", "prompt"])
            merged_df["formatted_predictions"] = merged_df["prediction"].apply(lambda x: extract_final_choice(x, ("A", "B", "C", "D", "E")))
        elif task == "emotion_reasoning":
            merged_df = df.merge(self.predictions_dataframe, on=["audio", "prompt"])
            merged_df["formatted_predictions"] = merged_df["prediction"].apply(lambda x: extract_final_choice(x, ("A", "B", "C", "D")))
        else:
            raise ValueError

        self._compute_accuracy(merged_df, pred_col="formatted_predictions", gt_col="correct_option")

        return merged_df

    def eval_event_duration_short_audio(self, offset=None):
        offset = offset or self.offset
        df = self.groundtruth_dataframe.copy()
        df["prompt"] = df["question"].apply(lambda x: self.create_duration_prompt(x, "short"))
        self.predictions_dataframe["prompt"] = self.predictions_dataframe["prompt"].apply(lambda x: remove_bracketed_tags(x.strip()))

        merged_df = df.merge(self.predictions_dataframe, on=["audio", "prompt"])
        self._compute_accuracy(merged_df, pred_col="prediction", gt_col="groundtruth", offset=offset)


    def eval_event_duration(self, offset=None):
        offset = offset or self.offset
        df = self.groundtruth_dataframe.copy()
        df["prompt"] = df["question"].apply(lambda x: self.create_duration_prompt(x, "long"))

        df_seconds = df[df.answer_type == "seconds"]
        df_minutes = df[df.answer_type == "minutes"]

        def clean_and_convert_groundtruth(x, to_minutes=False):
            x = str(x).strip().strip(".")
            return int(convert_minutes_to_seconds(x)) if to_minutes else int(x)

        def format_prediction_seconds(x):
            return int(convert_to_seconds(str(x).strip().strip(".")))

        def format_prediction_minutes(x):
            return int(convert_minutes_to_seconds(extract_integer(str(x).strip().strip("."))))

        merged_seconds = df_seconds.merge(self.predictions_dataframe, on=["audio", "prompt"])
        merged_minutes = df_minutes.merge(self.predictions_dataframe, on=["audio", "prompt"])

        merged_seconds["formatted_groundtruth"] = merged_seconds["groundtruth"].apply(clean_and_convert_groundtruth)
        merged_minutes["formatted_groundtruth"] = merged_minutes["groundtruth"].apply(
            lambda x: clean_and_convert_groundtruth(x, to_minutes=True)
        )

        merged_seconds["formatted_predictions"] = merged_seconds["prediction"].apply(format_prediction_seconds)
        merged_minutes["formatted_predictions"] = merged_minutes["prediction"].apply(format_prediction_minutes)

        combined_df = pd.concat([merged_seconds, merged_minutes], ignore_index=True)
        self._compute_accuracy(combined_df, pred_col="formatted_predictions", gt_col="formatted_groundtruth", offset=offset)



