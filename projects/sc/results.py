import logging
import ast
import re

from pydantic import BaseModel, ValidationError
from typing import Any, List
import csv
import pandas as pd
import os
from collections import Counter, defaultdict
from glob import glob
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import OmegaConf
from torchtune import config
from tqdm import tqdm

from research.utils.logging import setup_logger

log = logging.getLogger(__file__)


def str_to_list(s):
    return [float(x) for x in ast.literal_eval(s)]


def _plot_token_trajectories(logits, tokens, correct):
    # Plotting setup
    for idx, (logit_seq, token_seq, is_correct) in enumerate(zip(logits, tokens, correct)):
        log_probs = torch.log_softmax(
            torch.Tensor(logit_seq), dim=-1
        )  # assumed logits are np.ndarray; convert to tensor for computation
        log_prob_trajectory = log_probs.max(
            axis=-1
        ).values.numpy()  # Getting the max log prob for each token

        # Get the log probability trajectory for the sampled token sequence
        # Assume token_seq is a list of indices corresponding to the actual tokens
        # log_prob_trajectory = log_probs[torch.arange(len(token_seq)), token_seq].numpy()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=log_prob_trajectory, marker="o", label="Token Log Probs")
        plt.axhline(y=0, color="r", linestyle="--", linewidth=1, label="Base Log Prob Level")
        plt.title(
            f"Trajectory of Token Log Probs - {'Correct' if is_correct else 'Incorrect'} Example #{idx}"
        )
        plt.xlabel("Time step")
        plt.ylabel("Log Probability")
        plt.legend()

        # Customize this for saving or showing each plot
        plt.savefig(f"log_probs_trajectory_{idx}.png")
        plt.close()


def preprocessing(
    *,
    config_file: str = "projects/sc/configs/llama3_1_8b_config.yaml",
    dataset_id: str = "gsm8k",
    level="INFO",
):
    setup_logger(level)

    cfg = OmegaConf.load(config_file)
    model_id = cfg.model_id

    # get file list
    # f"{idx}_{rank}-{world_size}_{dataset_name}_results_{seed}.pt"
    file_list = glob(os.path.join("output", "features", model_id, f"*-8*{dataset_id}_results_*.pt"))
    log.debug(f"{file_list=}")

    log.info("Loading tokenizer")
    tokenizer = config.instantiate(cfg.tokenizer)

    results = defaultdict(list)
    for f in tqdm(file_list, desc="Joining results"):
        seed = int(f.split("_")[-1].replace(".pt", ""))
        res = torch.load(
            f,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

        res["seed"] = seed
        res["model_id"] = model_id
        res["dataset_id"] = dataset_id

        for k, v in res.items():
            if k == "tokens":
                results[k].append(v)

                # decode tokens response
                k = "model_answers"
                t = v.clone()
                v = []
                model_final_answers = []
                for tt in t:
                    text: str = tokenizer.decode(tt.numpy().tolist())
                    v.append(text)
                    # get model final answer
                    mfa = (
                        text.split("The answer is: ")[-1]
                        .split(".")[0]
                        .strip()
                        .replace("!", "")[:10]
                    )
                    model_final_answers.append(mfa)
                results["model_final_answer"].append(model_final_answers)

            elif k == "logits":
                pass

            elif k == "hidden":
                continue

            results[k].append(v)

    first = {k: v[42] for k, v in results.items()}

    log.info(f"first['model_answers']={first['model_answers']}")
    log.info(f"first['final_answer']={first['final_answer']}")
    log.info(f"first['model_final_answer']={first['model_final_answer']}")
    log.info(f"first['logits']={first['logits'][:, 1].max(-1).values}")
    log.info(f"{results.keys()=}")

    dest_dir = os.path.join("output", "preprocessing", "sc")
    os.makedirs(dest_dir, exist_ok=True)
    log.info("Saving...")
    torch.save(results, os.path.join(dest_dir, f"{model_id.replace("/", "_")}_{dataset_id}.pt"))
    log.info("Done!")
    return


def acc_preprocessing(
    *,
    config_file: str = "projects/sc/configs/llama3_1_8b_config.yaml",
    dataset_id: str = "gsm8k",
    level="INFO",
):
    setup_logger(level)

    cfg = OmegaConf.load(config_file)
    model_id = cfg.model_id
    dest_dir = os.path.join("output", "preprocessing", "sc")
    res = torch.load(
        os.path.join(dest_dir, f"{model_id.replace("/", "_")}_{dataset_id}.pt"),
        weights_only=False,
        mmap=True,
    )

    # results.keys()=dict_keys(['tokens', 'model_final_answer', 'model_answers', 'logits', 'prompt_len', 'max_len', 'temperature', 'top_p', 'top_k', 't_prefill', 't_total', 'mask', 'question', 'answer', 'prompt', 'final_answer', 'seed', 'model_id', 'dataset_id'])
    # Define the Pydantic model
    class DataModel(BaseModel):
        id: str
        final_answer: float
        model_final_answer: List[Any]
        model_id: str
        dataset_id: str
        seed: int

    # Function to transform a list of Pydantic objects into a CSV file
    def transform_to_csv(data_list: List[DataModel], csv_file_path: str):
        # Extract field names from the Pydantic model
        field_names = DataModel.__fields__.keys()

        # Open the CSV file for writing
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=field_names)

            # Write the header
            writer.writeheader()

            # Write the data rows
            for data in data_list:
                writer.writerow(data.dict())

    # Function to transform the results dictionary into a list of DataModel objects
    def transform_results_to_list(results: dict) -> List[DataModel]:
        data_list = []
        pattern = r"\d+\.?\d*"
        for i in tqdm(range(len(results["final_answer"]))):
            fa = float(results["final_answer"][i][0].replace(",", ""))

            mfa = []
            for text in results["model_final_answer"][i]:
                matches = re.findall(pattern, text)
                mfa.extend([float(match) for match in matches])

            data = DataModel(
                id=str(hash(results["question"][i][0])),
                final_answer=fa,
                model_final_answer=mfa,
                model_id=results["model_id"][i],
                dataset_id=results["dataset_id"][i],
                seed=results["seed"][i],
            )
            data_list.append(data)

        return data_list

    data_list = transform_results_to_list(res)
    transform_to_csv(data_list, "/tmp/output.csv")

    df = pd.read_csv("/tmp/output.csv")
    df["model_final_answer"] = df["model_final_answer"].apply(str_to_list)
    assert (
        df["model_final_answer"]
        .apply(lambda x: isinstance(x, list) and all(isinstance(i, float) for i in x))
        .all()
    )
    print(df.head(5))
    df = (
        df.groupby("id")
        .agg({
            "final_answer": "first",
            "model_final_answer": lambda x: sum(x, []),
            "model_id": "first",
            "dataset_id": "first",
            "seed": "first",
        })
        .reset_index()
    )

    def compute_accuracy(row):
        final_answer = row["final_answer"]
        model_answers = row["model_final_answer"]

        correct_count = sum(1 for answer in model_answers if answer == final_answer)
        total_count = len(model_answers)

        accuracy = (correct_count / total_count) * 100
        return accuracy

    # Apply the function to each row and create a new column 'accuracy'
    df["accuracy"] = df.apply(compute_accuracy, axis=1)

    dest_dir = os.path.join("output", "preprocessing", "sc")
    filename = os.path.join(dest_dir, f"{model_id.replace("/", "_")}_{dataset_id}_acc.csv")

    print(df)
    df.to_csv(filename, index=False)

    return


def viz():
    dest_dir = os.path.join("output", "preprocessing", "sc")
    file_list = glob(os.path.join(dest_dir, "*acc.csv"))
    dfs = [pd.read_csv(f) for f in file_list]
    df = pd.concat(dfs)
    models = df["model_id"].unique()
    datasets = df["dataset_id"].unique()
    df = df.query(f"model_id = '{models[0]}' and dataset_id = '{datasets[0]}'")
    x = df["final_answer"]
    y = df["accuracy"]
    plt.figure()
    sns.jointplot(x=x, y=y, kind="scatter", marginal_kws=dict(bins=20, fill=True))
    plt.title(f"{models[0]} - {datasets[0]}")
    plt.show()

    return
    # Results extraction
    hidden = res["hidden"]  # n, p, s, d
    logits = res["logits"]  # n, p, s, k
    tokens = res["tokens"]  # n, p, s
    answer = res["answer"]  # n, p
    gtruth = res["gtruth"]  # n

    # Step 1: Majority Vote of Answers
    majority_vote_answers = [Counter(answer_set).most_common(1)[0][0] for answer_set in answer]

    # Step 2: Calculate Accuracy
    correct = np.array(majority_vote_answers) == np.array(gtruth)
    accuracy = np.sum(correct) / len(gtruth)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Step 3: Plot the Tokens Log Probs Trajectory
    plot_token_trajectories(logits, tokens, correct)


if __name__ == "__main__":
    from fire import Fire

    Fire({
        "preprocessing": preprocessing,
        "acc_preprocessing": acc_preprocessing,
    })
