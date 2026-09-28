import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_qncd_runtime_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "svdquant"
        / "sdxl_w3a4_qncd"
        / "scripts"
        / "qncd_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("qncd_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qncd_renoise_uses_euler_ve_sigma_variance():
    module = _load_qncd_runtime_module()
    prev_latents = torch.tensor([[[[1.0, -2.0]]]])
    injected_noise = torch.tensor([[[[0.5, -1.5]]]])
    sigma = torch.tensor(3.0)
    sigma_next = torch.tensor(1.0)

    actual = module.renoise_euler_ve(prev_latents, sigma, sigma_next, injected_noise)
    expected = prev_latents + torch.sqrt(sigma**2 - sigma_next**2) * injected_noise

    assert torch.allclose(actual, expected)


class _DummyTokenizer:
    model_max_length = 4

    def __call__(self, prompt, **_kwargs):
        value = 7 if prompt else 0
        return SimpleNamespace(input_ids=torch.full((1, self.model_max_length), value, dtype=torch.long))


class _DummyEncoderOutput:
    def __init__(self, pooled, hidden):
        self.hidden_states = [hidden * 0, hidden, hidden * 2]
        self._pooled = pooled

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self._pooled


class _DummyEncoder:
    def __call__(self, input_ids, output_hidden_states=True):
        assert output_hidden_states
        value = input_ids.to(torch.float32).mean()
        hidden = torch.full((*input_ids.shape, 2), value + 1.0, dtype=torch.float32)
        pooled = torch.full((input_ids.shape[0], 2), value + 2.0, dtype=torch.float32)
        return _DummyEncoderOutput(pooled, hidden)


def test_qncd_conditioning_uses_zero_negative_for_pipeline_none_prompt():
    module = _load_qncd_runtime_module()

    prompt_embeds, add_text_embeds, add_time_ids, do_cfg = module.prepare_conditioning(
        "prompt",
        None,
        _DummyTokenizer(),
        _DummyTokenizer(),
        _DummyEncoder(),
        _DummyEncoder(),
        torch.device("cpu"),
        guidance_scale=7.5,
        force_zeros_for_empty_prompt=True,
    )

    assert do_cfg
    assert torch.count_nonzero(prompt_embeds[0]).item() == 0
    assert torch.count_nonzero(add_text_embeds[0]).item() == 0
    assert torch.count_nonzero(prompt_embeds[1]).item() > 0
    assert torch.count_nonzero(add_text_embeds[1]).item() > 0
    assert torch.equal(add_time_ids[0], add_time_ids[1])
