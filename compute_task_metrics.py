import argparse
import logging

from src.metrics.all_tasks_metrics import LocalizationMetric, NonLocalizationMetric

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TASKS = [
    "word_localization",
    "advertisement_localization",
    "named_entity_localization",
    "speaker_number_estimation",
    "entire_duration",
    "event_duration",
    "event_duration_short_audio",
    "emotion_ranking",
    "emotion_reasoning",
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate any task")
    parser.add_argument('--task', type=str, required=True, help="Task to be evaluated")
    parser.add_argument('--groundtruth_file', required=True, type=str, help="Groundtruth file path")
    parser.add_argument('--predictions_file', required=True, type=str, help="Predictions file path")
    parser.add_argument('--offset', type=int, default=2 , help="Tolerance (in seconds) to be applied for duration-based to allow for small timing prediction discrepancies")

    args = parser.parse_args()

    if args.task not in  TASKS:
        raise ValueError(f"This script doesn't support '{args.task}'")

    logging.info(f"Evaluating task: {args.task}")

    if "localization" in args.task:
        metric = LocalizationMetric(groundtruth_filepath=args.groundtruth_file,
                                    predictions_filepath=args.predictions_file,
                                    task=args.task)
    else:
        metric_kwargs = {
            "groundtruth_json": args.groundtruth_file,
            "predictions_json": args.predictions_file,
            "task": args.task
        }
        if "emotion" in args.task:
            metric_kwargs["prompt_suffix"] = (
                "Listen to the audio and select one option from the provided choices "
                "that best matches the answer"
            )
        if "duration" in args.task:
            metric_kwargs["offset"] = args.offset

        metric = NonLocalizationMetric(**metric_kwargs)

    logging.info(f"Metrics are: {metric.metrics}")











