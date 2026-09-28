# -*- coding: utf-8 -*-
"""Language model evaluator for LongBench."""

import json
import os
import typing as tp

import numpy as np
import torch
import torch.utils.data
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from deepcompressor.utils import tools

from ..base import LlmEvaluatorBase
from .metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)

__all__ = ["LongbenchEvaluator"]


class LongbenchEvaluator(LlmEvaluatorBase):
    task2maxlen: dict[str, int] = {
        "narrativeqa": 128,
        "qasper": 128,
        "multifieldqa_en": 64,
        "multifieldqa_zh": 64,
        "hotpotqa": 32,
        "2wikimqa": 32,
        "musique": 32,
        "dureader": 128,
        "gov_report": 512,
        "qmsum": 512,
        "multi_news": 512,
        "vcsum": 512,
        "trec": 64,
        "triviaqa": 32,
        "samsum": 128,
        "lsht": 64,
        "passage_count": 32,
        "passage_retrieval_en": 32,
        "passage_retrieval_zh": 32,
        "lcc": 64,
        "repobench-p": 64,
    }

    task2prompt: dict[str, str] = None

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        eos_token_ids: tp.Sequence[int],
        output_dirpath: str = "",
        task2maxlen: dict[str, int] = None,
        task2prompt: dict[str, str] = None,
    ):
        super().__init__(model=model, tokenizer=tokenizer)
        self.model_name = model_name
        self.eos_token_ids = eos_token_ids
        if task2maxlen is not None:
            self.task2maxlen = task2maxlen
        if task2prompt is not None:
            self.task2prompt = task2prompt
        self.output_dirpath = output_dirpath
        self.logger = tools.logging.getLogger(__name__)

    def filter_tasks(self, tasks: list[str]) -> list[str]:
        """Filter the tasks to only include supported tasks."""
        if "longbench-e" in tasks:
            return ["longbench-e"]
        if "longbench" in tasks:
            return sorted(self.task2maxlen.keys(), key=lambda x: self.task2maxlen[x])
        return sorted([task for task in tasks if task in self.task2maxlen], key=lambda x: self.task2maxlen[x])

    def evaluate(self, tasks: list[str], max_length: int, **kwargs) -> dict[str, dict[str, dict[str, float]]]:
        """Evaluate the model on the given tasks."""
        ...
        tools.logging.Formatter.indent_inc()
        longbench_e = False
        if "longbench-e" in tasks:
            assert len(tasks) == 1, "LongBench-E should be the only task"
            longbench_e = True
            tasks = [
                "hotpotqa",
                "2wikimqa",
                "triviaqa",
                "passage_count",
                "multifieldqa_en",
                "trec",
                "lcc",
                "repobench-p",
                "qasper",
                "samsum",
                "gov_report",
                "multi_news",
                "passage_retrieval_en",
            ]
        result = {"results": {}, "versions": {}}
        for task in tasks:
            self.logger.info(f"- Evaluating on {task}")
            tools.logging.Formatter.indent_inc()
            preds = self.predict(task=task, max_length=max_length)
            if not preds:
                self.logger.warning(f"No results for {task}")
                tools.logging.Formatter.indent_dec()
                continue
            if self.output_dirpath:
                self.logger.info(f"+ Saving results for {task} to {self.output_dirpath}")
                os.makedirs(os.path.join(self.output_dirpath, "longbench"), exist_ok=True)
                with open(
                    os.path.join(self.output_dirpath, "longbench", f"{task}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for pred in preds:
                        json.dump(pred, f, ensure_ascii=False)
                        f.write("\n")
            predictions, answers, lengths = [], [], []
            for pred in preds:
                predictions.append(pred["prediction"])
                answers.append(pred["answers"])
                lengths.append(pred["length"])
            all_classes = preds[0]["all_classes"]
            if longbench_e:
                scores = LongbenchScorer.scorer_e(
                    task=task,
                    predictions=predictions,
                    answers=answers,
                    lengths=lengths,
                    all_classes=all_classes,
                )
            else:
                scores = {
                    "score": LongbenchScorer.score(
                        task=task,
                        predictions=predictions,
                        answers=answers,
                        all_classes=all_classes,
                    )
                }
            tools.logging.debug(f"+ Scores: {scores}", self.logger)
            result["results"][task] = scores
            result["versions"][task] = 1
            tools.logging.Formatter.indent_dec()
        tools.logging.Formatter.indent_dec()
        return result

    def predict(
        self,
        task: str,
        max_length: int,
        max_gen_length: int | None = None,
        prompt_format: str = "",
    ) -> list[dict[str, tp.Any]]:
        if max_gen_length is None:
            max_gen_length = self.task2maxlen[task]
        if prompt_format == "":
            prompt_format = self.task2prompt[task]
        dataset = load_dataset("THUDM/LongBench", task, split="test")
        preds = []
        pbar = tqdm(dataset)
        tools.logging.Formatter.indent_inc()
        for idx, data in enumerate(pbar):
            prompt = prompt_format.format(**data)
            # truncate to fit max_length
            # (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = self.tokenizer.decode(
                    tokenized_prompt[:half], skip_special_tokens=True
                ) + self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if task not in ("trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"):
                # chat models are better off without build prompts on these tasks
                prompt = self.build_chat(prompt)
            input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            pbar.set_description(f"Generating for {idx}, len={input.input_ids.shape[-1]}")
            with torch.no_grad():
                output = self.model(input_ids=input.input_ids, past_key_values=None, use_cache=True)
                past_key_values = output.past_key_values
                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content = [pred_token_idx.item()]
                for _ in range(max_gen_length - 1):
                    outputs = self.model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_content += [pred_token_idx.item()]
                    if pred_token_idx.item() in self.eos_token_ids:
                        break
            pred = self.tokenizer.decode(generated_content, skip_special_tokens=True)
            pred = self.post_process(pred)
            # tools.logging.debug(f"- Prediction: {pred}", self.logger)
            preds.append(
                {
                    "prediction": pred,
                    "answers": data["answers"],
                    "all_classes": data["all_classes"],
                    "length": data["length"],
                }
            )
            # break
        tools.logging.Formatter.indent_dec()
        return preds

    def build_chat(self, prompt):
        """Build chat prompt for chat models."""
        if "llama-2" in self.model_name:
            prompt = f"[INST]{prompt}[/INST]"
        return prompt

    def post_process(self, response: str) -> str:
        if "xgen" in self.model_name:
            response = response.strip().replace("Assistant:", "")
        elif "internlm" in self.model_name:
            response = response.split("<eoa>")[0]
        elif "llama-3" in self.model_name:
            response = response.split(".assistant")[0].split("\n\nQuestion")[0].split("</s>")[0].strip()
        elif "llama-2-7b" in self.model_name and "instruct" in self.model_name and "32k" in self.model_name:
            response = (
                response.split("(Document")[0]
                .split("\n\nQuestion")[0]
                .split("\n\nAnswer")[0]
                .split("(Passage")[0]
                .strip()
            )
        return response


class LongbenchScorer:
    task2metric = {
        "narrativeqa": qa_f1_score,
        "qasper": qa_f1_score,
        "multifieldqa_en": qa_f1_score,
        "multifieldqa_zh": qa_f1_zh_score,
        "hotpotqa": qa_f1_score,
        "2wikimqa": qa_f1_score,
        "musique": qa_f1_score,
        "dureader": rouge_zh_score,
        "gov_report": rouge_score,
        "qmsum": rouge_score,
        "multi_news": rouge_score,
        "vcsum": rouge_zh_score,
        "trec": classification_score,
        "triviaqa": qa_f1_score,
        "samsum": rouge_score,
        "lsht": classification_score,
        "passage_retrieval_en": retrieval_score,
        "passage_count": count_score,
        "passage_retrieval_zh": retrieval_zh_score,
        "lcc": code_sim_score,
        "repobench-p": code_sim_score,
    }

    @staticmethod
    def score(
        task: str,
        predictions: tp.Sequence[str],
        answers: tp.Sequence[tp.Sequence[str]],
        all_classes: tp.Sequence[str],
        task2metric: tp.Mapping[str, tp.Callable[[str, str, tp.Any], float]] = None,
    ) -> float:
        if task2metric is None:
            task2metric = LongbenchScorer.task2metric
        total_score = 0.0
        for prediction, ground_truths in zip(predictions, answers, strict=True):
            score = 0.0
            prediction = (
                prediction.split(".assistant")[0]
                .split("\n\nQuestion")[0]
                .split("</s>")[0]
                .split("(Document")[0]
                .split("\n\nQuestion")[0]
                .split("\n\nAnswer")[0]
                .split("(Passage")[0]
                .strip()
            )
            if task in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip("\n").split("\n")[0]
            if task in ["multifieldqa_zh", "dureader"]:
                prediction = prediction.split("问题：")[0].strip()
            if task in ["lsht"]:
                prediction = prediction.split("新闻内容：")[0].strip()
            if task in ["passage_retrieval_zh"]:
                prediction = prediction.split("请问")[0].split("提示")[0].strip()
            for ground_truth in ground_truths:
                score = max(
                    score,
                    task2metric[task](prediction, ground_truth, all_classes=all_classes),
                )
            total_score += score
        return round(100 * total_score / len(predictions), 2)

    @staticmethod
    def scorer_e(
        task: str,
        predictions: tp.Sequence[str],
        answers: tp.Sequence[tp.Sequence[str]],
        lengths: tp.Sequence[int],
        all_classes: tp.Sequence[str],
        task2metric: tp.Mapping[str, tp.Callable[[str, str, tp.Any], float]] = None,
    ) -> dict[str, float]:
        if task2metric is None:
            task2metric = LongbenchScorer.task2metric
        scores = {"0-4k": [], "4-8k": [], "8k+": []}
        for prediction, ground_truths, length in zip(predictions, answers, lengths, strict=True):
            score = 0.0
            if task in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip("\n").split("\n")[0]
            for ground_truth in ground_truths:
                score = max(
                    score,
                    task2metric[task](prediction, ground_truth, all_classes=all_classes),
                )
            if length < 4000:
                scores["0-4k"].append(score)
            elif length < 8000:
                scores["4-8k"].append(score)
            else:
                scores["8k+"].append(score)
        for key in scores.keys():
            scores[key] = round(100 * np.mean(scores[key]), 2)
        return scores


# Initialize the evaluator task2prompt by loading the json file
with open(os.path.join(os.path.dirname(__file__), "task2prompt.json")) as f:
    LongbenchEvaluator.task2prompt = json.load(f)
