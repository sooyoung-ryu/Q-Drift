# -*- coding: utf-8 -*-
"""Language model customized evaluator."""

import math

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import LlmEvaluatorBase

__all__ = ["LlmCustomEvaluator"]


class LlmCustomEvaluator(LlmEvaluatorBase):
    def filter_tasks(self, tasks: list[str]) -> list[str]:
        """Filter the tasks to only include supported tasks."""
        return [task for task in tasks if task.startswith(("wikitext", "pile"))]

    def evaluate(
        self, tasks: list[str], max_length: int | None = None, **kwargs
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Evaluate the model on the given tasks.

        Args:
            tasks (`list[str]`): List of tasks to evaluate on.
            max_length (`int`, optional, defaults to `None`): Maximum length for the model.

        Returns:
            dict[str, dict[str, dict[str, float]]]: Evaluation results `{"results": {"task": {"metric": score}}}`.
        """
        result = {"results": {}, "versions": {}}
        for task in tasks:
            result["results"][task] = {
                "word_perplexity": _eval_ppl_with_gptq_evaluator(
                    self.model, self.tokenizer, task=task, seq_length=max_length
                )
            }
            result["versions"][task] = 1
        return result


def _eval_ppl_with_gptq_evaluator(
    model: PreTrainedModel,
    /,
    tokenizer: PreTrainedTokenizer,
    task: str,
    seq_length: int = 2048,
    max_num_samples: int = -1,
) -> float:
    """Evaluate the perplexity of a model on a task using GPTQ style evaluation.

    Args:
        model (`PreTrainedModel`):
            The model.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer.
        task (`str`):
            The task name.
        seq_length (`int`, *optional*, defaults to `2048`):
            The sequence length.
        max_num_samples (`int`, *optional*, defaults to `-1`):
            The maximum number of samples to evaluate.

    Returns:
        float: The perplexity.
    """
    assert seq_length > 0, "seq_length must be positive"
    if task.startswith("wikitext"):
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_dataset = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    elif task.startswith("pile"):
        test_dataset = load_dataset("pile", task, split="test")
        test_dataset = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    else:
        raise ValueError(f"Invalid task: {task}")

    test_dataset = test_dataset.input_ids.to(model.device)
    num_samples = test_dataset.numel() // seq_length
    if max_num_samples > 0:
        num_samples = min(num_samples, max_num_samples)
    model = model.eval()

    nlls = []
    for i in tqdm(range(num_samples), desc=f"evaluating on {task} with seq_length {seq_length}", dynamic_ncols=True):
        batch = test_dataset[:, (i * seq_length) : ((i + 1) * seq_length)]
        with torch.inference_mode():
            shift_logits = model(batch.to(model.device)).logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seq_length
        nlls.append(neg_log_likelihood)
    return math.exp(sum(nlls) / (num_samples * seq_length))
