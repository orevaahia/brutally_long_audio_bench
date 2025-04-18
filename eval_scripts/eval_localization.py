import argparse
import logging
import os

from datetime import datetime
from utils.eval_utils import load_json, save_json
from utils.span_localization import evaluate_span_localization
from utils.word_localization import  evaluate_word_localization



def main():
    parser = argparse.ArgumentParser(description="Evaluate the localization tasks")
    parser.add_argument('--task', type=str, help="Task to be evaluated. word, nel, ads")
    parser.add_argument('--groundtruth_file', type=str, help="Json file with groundtruth timestamps")
    parser.add_argument('--predictions_file', type=str, help="Json file with predicted timestamps")
    parser.add_argument('--output_folder', type=str, help="Output folder")

    args = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_file = f"results_{os.path.basename(args.predictions_file)}_{current_time}"
    output_folder = os.path.join(args.output_folder, args.task)
    os.makedirs(output_folder, exist_ok=True)

    output_metrics_file = os.path.join(output_folder, metrics_file)


    output_metrics_file = f"results_{current_time}_{os.path.basename(args.predictions_file)}"

    if args.task == "word_localization":
        metrics_output = evaluate_word_localization(args.predictions_file, args.groundtruth_file)
    elif args.task == "advertisement_localization":
        metrics_output  = evaluate_span_localization(args.predictions_file, args.groundtruth_file, "advertisement")
    elif args.task == "ne_localization":
        metrics_output  = evaluate_span_localization(args.predictions_file, args.groundtruth_file, "entity")

    else:
        raise ValueError(f"Unsupported task: {args.task}. Supported tasks are: word_localization, advertisement_localization, ne_localization.")

    # Save metrics
    save_json(metrics_output, output_metrics_file)

    logging.info(f"Saved metrics {output_metrics_file}")



if __name__ == '__main__':
    main()
