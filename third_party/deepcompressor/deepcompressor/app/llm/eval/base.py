# -*- coding: utf-8 -*-
"""Language model evaluator base."""

from abc import ABC, abstractmethod

from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = ["LlmEvaluatorBase"]


class LlmEvaluatorBase(ABC):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model, self.tokenizer = model, tokenizer

    @abstractmethod
    def filter_tasks(self, tasks: list[str]) -> list[str]:
        """Filter the tasks to only include supported tasks."""
        ...

    @abstractmethod
    def evaluate(self, tasks: list[str], **kwargs) -> dict[str, dict[str, dict[str, float]]]:
        """Evaluate the model on the given tasks."""
        ...
