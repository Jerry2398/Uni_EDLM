# Dataset

This is the introduction of the data we used, and how you can download, preprocess, and modify your own evaluation and training data.

## Download Original Data

By default, datasets are saved to `Your Dataset Path`. You can use a custom directory with `--output-dir`.

Download data (from the `data/` directory):

```bash
cd data

# download the evaluation data
python download_data.py --dataset MATH500
python download_data.py --dataset GSM8K
python download_data.py --dataset AIME2024
python download_data.py --dataset LiveCodeBench
python download_data.py --dataset LiveBench
python download_data.py --dataset MBPP
python download_data.py --dataset HumanEval
# download the training data
## rl data
python download_data.py --dataset MATH_train
python download_data.py --dataset PrimeIntellect
## sft data
python download_data.py --dataset demon_openr1math
```

### Using datasets from a custom directory

To use datasets from your own directory (e.g. `Your Dataset Path`), set `dataset.data_dir` in your config YAML so that training and evaluation scripts load the JSON files from there.

Example (add under the `dataset:` section in your config, e.g. `configs/trado_eval.yaml` or `configs/sft_trado.yaml`):

```yaml
dataset:
    data_dir: "Your Dataset Path"
    eval_dataset: "MATH500"   # or optimization_data for SFT/RL
    data_type: "math"
```

If `data_dir` is not set, the code uses `./data` (when running from project root) or `../data` (when running from `sample/`).


## Preprocess the SFT data

RL data can be used directly, but you should preprocess the SFT data (such like `demon_openr1math`, rl data is not needed) first by `preprocess_sft.ipynb`.

## For multimodal data

We categorize data into two types: **SFT Data** (Supervised Fine-Tuning) and **RL/Eval Data** (Reinforcement Learning & Evaluation).

### Datasets
*   **SFT**: `data/sft_test_v.json` (provided as an example).
*   **RL & Eval**: `data/SEEDBench_IMG_32.json` (provided as an example).

### Image Directory
All images used in these datasets should be stored in a unified directory, e.g., `data/images/`.
*   **Configuration**: In your YAML config files, set `image_root` to this directory path.
*   **Usage**: The system will automatically join `image_root` with the image paths defined in the dataset JSON files.

---


## Multiple-choice benchmarks (MMLU, MMLU-Pro, HellaSwag, ARC-C, GPQA)

To evaluate on MMLU, MMLU-Pro, HellaSwag, ARC-Challenge, or GPQA, download and convert them to the project JSON format (one file per benchmark) with:

```bash
cd data
pip install datasets   # if not already installed
python download_mc_benchmarks.py --output-dir Your Dataset Path
```

To download only specific benchmarks:

```bash
python download_mc_benchmarks.py --output-dir Your Dataset Path --benchmark MMLU --benchmark HellaSwag --benchmark ARC_C --benchmark GPQA --benchmark MMLUPro
```

Output files are written to the given `--output-dir` (default: `Your Dataset Path`):

| Benchmark   | File name     | Use as `eval_dataset` |
|------------|---------------|------------------------|
| MMLU       | `MMLU.json`   | `MMLU`                 |
| MMLU-Pro   | `MMLUPro.json`| `MMLUPro`              |
| HellaSwag  | `HellaSwag.json` | `HellaSwag`         |
| ARC-Challenge | `ARC_C.json` | `ARC_C`                |
| GPQA Diamond | `GPQA.json`  | `GPQA`                 |

In your eval config, set `dataset.data_type: "option"` and `dataset.eval_dataset` to the name in the table (e.g. `MMLU`). Use the same `dataset.data_dir` as above so the loader finds the JSON files.

## Customize Your Own Dataset

Your JSON dataset must contain the following necessary fields to perform both optimization and evaluation.

### Math or general tasks
1. `question`: This is the task description.
2. `ground_truth_answer`: This is the final ground-truth answer.

### Coding tasks

1. `question`: This is the coding task description.
2. `test_time_limit`: The time limit for each task's execution, usually set as 1.
3. `test_method`: `function` or `stdio` for each task.

For different test methods, we have different entries. For `function` format:

4. `test_list` use `assert` to test, example:
```python
assert my_sum(1, 2) == 3
```
5. `Prefix`: for `MBPP` and `HumanEval`, we use the prefix provided in their original data.

for `stdio` format:

4. `test_input` and `test_output` are two lists of the same length, where each corresponding element represents the Stdio-format input and output, respectively, for example:
```python
"test_input": [
  "1\n2\n",
  "3\n4\n"
],
"test_output": [
  "3\n",
  "7\n"
]
```



The original sources for the datasets used in this paper are [LiveBench](https://huggingface.co/datasets/livebench/coding), [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite), and [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp).
