# -*- coding: utf-8 -*-
"""Language model evaluation config."""

import random
import typing as tp
from dataclasses import dataclass, field

import numpy as np
import omniconfig
import torch
from omniconfig import configclass
from transformers import PreTrainedModel, PreTrainedTokenizer

from deepcompressor.utils import tools

from .custom import LlmCustomEvaluator
from .lm_eval import LmevalEvaluator
from .longbench import LongbenchEvaluator

__all__ = ["LlmEvalConfig"]


@configclass
@dataclass
class LlmEvalConfig:
    """Large language model evaluation configuration.

    Attributes:
        num_gpus (`int`, *optional*, defaults to `1`):
            The number of GPUs to use.
        batch_size (`int`, *optional*, defaults to `1`):
            The batch size used for inference.
        tasks (`list[str]`, *optional*, defaults to `["zero-shot"]`):
            Task names, e.g. wikitext, hellaswag, piqa, winogrande.
        max_seq_length (`int`, *optional*, defaults to `-4096`):
            Maximum sequence length.
            If negative, sequence lengths smaller than or equal to the absolute value are used.
        evaluators (`list[str]`, *optional*, defaults to `["gptq"]`):
            Evaluators names.
        num_shot (`int`, *optional*, defaults to `None`):
            The number of shots for few-shot evaluation.
        fewshot_as_multiturn (`bool`, *optional*, defaults to `False`):
            Whether to treat few-shot evaluation as multi-turn.
        apply_chat_template (`bool`, *optional*, defaults to `False`):
            Whether to apply chat template for evaluation.
    """

    num_gpus: int = field(default=1, metadata={omniconfig.ARGPARSE_ARGS: ("--num-gpus", "-n")})
    batch_size: int = 1
    tasks: list[str] = field(
        default_factory=lambda: ["zero-shot"],
        metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": str}},
    )
    max_seq_length: int = -4096
    evaluators: list[str] = field(
        default_factory=lambda: ["gptq"], metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": str}}
    )
    num_shot: int | None = None
    fewshot_as_multiturn: bool = False
    apply_chat_template: bool = False

    def __post_init__(self):
        if "zero-shot" in self.tasks:
            self.tasks.remove("zero-shot")
            self.tasks.extend(("wikitext", "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge"))
        self.tasks = sorted({tast.lower() for tast in self.tasks})
        self.evaluators = sorted({evaluator.lower() for evaluator in self.evaluators})
        for evaluator in self.evaluators:
            assert evaluator in ("lm_eval", "gptq", "longbench"), f"Invalid evaluator: {evaluator}"
        if len(self.evaluators) == 1 and self.evaluators[0] == "gpq":
            self.tasks = [task for task in self.tasks if task.startswith(("wikitext", "pile", "gsm8k"))]
            assert len(self.tasks) > 0, "No valid tasks for GPTQ evaluation"

    def evaluate(
        self,
        model: PreTrainedModel,
        /,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        eos_token_ids: tp.Sequence[int] = (),
        output_dirpath: str = "",
    ) -> dict[str, dict[int, dict[str, dict[tp.Any, dict[str, tp.Any]]]]]:
        """Evaluate the model.

        Args:
            model (`PreTrainedModel`):
                The model.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer.
            model_name (`str`):
                The name of the model.
            eos_token_ids (`Sequence[int]`, *optional*, defaults to `()`):
                The EOS token IDs.

        Returns:
            `dict[str, dict[int, dict[str, dict[tp.Any, dict[str, tp.Any]]]]]`:
                The evaluation results.
                    - The first key is the evaluator name.
                    - The second key is the maximum sequence length.
                    - The third key is the content name, e.g., "results", "versions", "config".
                    - The fourth key is the task name for "results".
        """
        logger = tools.logging.getLogger(f"{__name__}.LlmEval")
        tools.logging.Formatter.indent_inc()
        tools.logging.Formatter.indent_dec()
        lm_max_seq_length = get_max_seq_length(model, tokenizer)
        max_seq_lengths = {2048, 4096, lm_max_seq_length}
        if self.max_seq_length < 0:
            if self.max_seq_length == -1:
                max_seq_length = lm_max_seq_length
            else:
                max_seq_length = min(lm_max_seq_length, -self.max_seq_length)
            max_seq_lengths = [length for length in sorted(max_seq_lengths) if length <= max_seq_length]
        elif self.max_seq_length == 0:
            max_seq_lengths = [lm_max_seq_length]
        else:
            max_seq_lengths = [self.max_seq_length]
        results = {}
        for evaluator_name in self.evaluators:
            logger.info(f"- Evaluator: {evaluator_name}")
            tasks = list(self.tasks)
            if evaluator_name == "gptq":
                evaluator = LlmCustomEvaluator(model=model, tokenizer=tokenizer)
            elif evaluator_name == "lm_eval":
                evaluator = LmevalEvaluator(model=model, tokenizer=tokenizer, batch_size=self.batch_size)
            elif evaluator_name == "longbench":
                evaluator = LongbenchEvaluator(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    eos_token_ids=eos_token_ids,
                    output_dirpath=output_dirpath,
                )
            else:
                raise ValueError(f"Invalid evaluator: {evaluator_name}")
            logger.info(f"- Tasks: {tasks}")
            logger.info(f"- Batch_size: {self.batch_size}")
            rsts = {}
            tools.logging.Formatter.indent_inc()
            for max_seq_length in max_seq_lengths:
                logger.info(f"+ Max_seq_length: {max_seq_length}")
                tools.logging.Formatter.indent_inc()
                tools.logging.Formatter.indent_inc()
                # set seed
                torch.manual_seed(42)
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                np.random.seed(42)
                random.seed(42)
                # evaluate
                rst = evaluator.evaluate(
                    tasks=tasks,
                    max_length=max_seq_length,
                    num_shot=self.num_shot,
                    fewshot_as_multiturn=self.fewshot_as_multiturn,
                    apply_chat_template=self.apply_chat_template,
                )
                rst["model"] = model_name
                tools.logging.Formatter.indent_dec()
                logger.info("- Results:")
                tools.logging.Formatter.indent_inc()
                tools.logging.info(self.make_table(rst), logger=logger)
                tools.logging.Formatter.indent_dec()
                rsts[max_seq_length] = rst
                tools.logging.Formatter.indent_dec()
            tools.logging.Formatter.indent_dec()
            results[evaluator_name] = rsts
        return results

    @staticmethod
    def make_table(rst: dict[str, dict[tp.Any, dict[str, tp.Any]]]) -> str:
        """Generate table of results.

        Args:
            results (`dict[str, dict[tp.Any, dict[str, tp.Any]]]`):
                The evaluation results.

        Returns:
            `str`:
                The string representation of the results in a table.
        """
        from pytablewriter import MarkdownTableWriter

        md_writer = MarkdownTableWriter()
        md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
        values = []
        for k, dic in rst["results"].items():
            version = rst["versions"][k]
            for m, v in dic.items():
                if "_stderr" in m:
                    continue
                mse = "_stderr,".join(m.split(","))
                appended = False
                if mse in dic:
                    se = dic[mse]
                    if isinstance(se, (int, float)):
                        values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
                        appended = True
                if not appended and isinstance(v, (int, float)):
                    values.append([k, version, m, "%.4f" % v, "", ""])
                    k = ""
                    version = ""
        md_writer.value_matrix = values
        return md_writer.dumps()


def get_max_seq_length(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, default_seq_length: int = 2048) -> int:
    seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
    for attr in seqlen_config_attrs:
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length == 1000000000000000019884624838656:
            return default_seq_length
        return tokenizer.model_max_length
    return default_seq_length
