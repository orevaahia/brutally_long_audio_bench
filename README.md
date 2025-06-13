# BLAB: Brutally Long Audio Bench

This repo contains the evaluation code for BLAB benchmark for the paper [BLAB: Brutally Long Audio Bench](https://arxiv.org/abs/2505.03054)


<p align="center">
Download BLAB Dataset  on Huggingface <a href="https://huggingface.co/datasets/oreva/blab_long_audio">🤗</a> <br>
Paper <a href="https://arxiv.org/abs/2505.03054"> 📑</a> ｜ Code <a href="https://github.com/orevaahia/brutally_long_audio_bench/"> ⚙️</a>
<br>
<a href="##Contact">Contact us</a><br>
<br>
</p>



## Brutally Long Audio Bench

We introduce Brutally Long Audio Bench (BLAB), a challenging long-form audio benchmark that evaluates audio LMs on localization, duration estimation, emotion, and counting tasks using audio segments averaging 51 minutes in length. BLAB consists of 833+ hours of diverse, full-length audio clips, each paired with human-annotated, text-based natural language questions and answers. Our audio data were collected from permissively licensed sources and underwent a human-assisted filtering process to ensure task compliance. We evaluate six open-source and proprietary audio LMs on BLAB and find that all of them, including advanced models such as Gemini 2.0 Pro and GPT-4o, struggle with the tasks in BLAB. Our comprehensive analysis reveals key insights into the trade-offs between task difficulty and audio duration. In general, we find that audio LMs struggle with long-form speech, with performance declining as duration increases. They perform poorly on localization, temporal reasoning, counting, and struggle to understand non-phonemic information, relying more on prompts than audio content. BLAB serves as a challenging evaluation framework to develop audio LMs with robust long-form audio understanding capabilities.


NB: This data should only be used for evaluation purposes and not for model training.



## Tasks Covered in BLAB

### Localization
* **Word Localization:** Locate the exact start and end times of specific words within the audio.
* **Named Entity Localization:** Detect and locate the exact start and end times of named entities (e.g., people, organizations, locations).
* **Advertisement Localization:** Locate and transcribe advertisement segments within a podcast.

### Counting
* **Speaker Number Estimation:** Determine the number of unique speakers present in the full audio segment.

### Duration
* **Event Duration:** Calculate the duration of specific acoustic events (e.g., laughter in a comedy special, question-and-answer segments in a panel session, or a particular speaker’s total speaking time in a meeting) within an audio sample,.
* **Entire Duration:** Estimate the total duration of an audio file, expressed in seconds.

### Emotion
* **Emotion Reasoning:** Reason over emotional expressions conveyed in the audio.
* **Emotion Ranking:** Rank different emotional expressions of speech and non-verbal sound present in the audio.


## Download and Access Data
The data can be downloaded on [huggingface]("https://huggingface.co/datasets/oreva/blab_long_audio).

```python
from huggingface_hub import snapshot_download

# Download data for all tasks
snapshot_download(repo_id="oreva/blab_long_audio", repo_type="dataset", allow_patterns="*.json", local_dir=".")
```

## Evaluation
* We've provided the **exact prompts used for evaluation** in the appendix of our accompanying paper.
* You can use the [compute_task_metrics.py](compute_task_metrics.py) script, to **evaluate your model's predictions** for BLAB.
* To use the evaluation script, you'll need to provide a JSON file containing a list of your model's predictions and specify your task. Each entry in this list should follow this structure, for each specific task:

    ```json
    [
      {
        "audio": "audio_1",
        "prediction": "model_prediction_for_audio_1",
        "prompt": "prompt_used_for_audio_1"
      },
     // ... more predictions
    ]
    ```


```bash
# Sample script usage
python compute_task_metrics.py \
    --task word_localization \
    --predictions_file predictions.json \
    --groundtruth_file blab_long_audio/word_localization.json \
   # --offset 2 (Tolerance of 2 seconds applied to only duration tasks. )

# Extracting structured data from language model outputs, especially in JSON format, is still challenging.
# While our evaluation script tries to accommodate various output styles, some model predictions may not be perfectly parseable and this might result in errors.
```





## Contact
If you have any questions, please feel free to contact us via oahia@cs.washington.edu .


## Citation

```
@misc{ahia2025blabbrutallylongaudio,
      title={BLAB: Brutally Long Audio Bench},
      author={Orevaoghene Ahia and Martijn Bartelds and Kabir Ahuja and Hila Gonen and Valentin Hofmann and Siddhant Arora and Shuyue Stella Li and Vishal Puttagunta and Mofetoluwa Adeyemi and Charishma Buchireddy and Ben Walls and Noah Bennett and Shinji Watanabe and Noah A. Smith and Yulia Tsvetkov and Sachin Kumar},
      year={2025},
      eprint={2505.03054},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.03054},
}

```