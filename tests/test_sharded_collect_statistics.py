import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PIXART_PATH = ROOT / "experiments" / "svdquant" / "pixart-sigma_w3a4" / "scripts" / "collect_statistics.py"
SANA_PATH = ROOT / "experiments" / "svdquant" / "sana_w3a4" / "scripts" / "collect_statistics.py"


def _install_import_stubs():
    diffusers = types.ModuleType("diffusers")

    class _DummyScheduler:
        @classmethod
        def from_config(cls, *args, **kwargs):
            return cls()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _DummyPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    diffusers.DPMSolverMultistepScheduler = _DummyScheduler
    diffusers.PixArtSigmaPipeline = _DummyPipeline
    diffusers.SanaPipeline = _DummyPipeline
    sys.modules.setdefault("diffusers", diffusers)

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *args, **kwargs: ""
    sys.modules.setdefault("huggingface_hub", hub)

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda *args, **kwargs: types.SimpleNamespace(update=lambda *_: None, close=lambda: None)
    sys.modules.setdefault("tqdm", tqdm_mod)

    pix_loader = types.ModuleType("deepcompressor_pixart_loader")
    pix_loader.apply_deepcompressor_patches = lambda *args, **kwargs: None
    pix_loader.load_quant_transformer = lambda *args, **kwargs: None
    sys.modules.setdefault("deepcompressor_pixart_loader", pix_loader)

    sana_loader = types.ModuleType("deepcompressor_sana_loader")
    sana_loader.apply_deepcompressor_patches = lambda *args, **kwargs: None
    sana_loader.load_quant_transformer = lambda *args, **kwargs: None
    sys.modules.setdefault("deepcompressor_sana_loader", sana_loader)

    yaml_mod = types.ModuleType("yaml")
    yaml_mod.safe_load = lambda *_args, **_kwargs: {}
    sys.modules.setdefault("yaml", yaml_mod)


def _load_module(path: Path, name: str):
    _install_import_stubs()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _exercise_module(module):
    assert [module.rank_index_range(10, rank, 3) for rank in range(3)] == [(0, 4), (4, 7), (7, 10)]

    with tempfile.TemporaryDirectory() as tmp:
        settings = {"num_inference_steps": 20, "guidance_scale": 4.5}
        collector = module.OutputCollector()
        collector.add_pair(1, torch.ones(1, 2, 2, 2), torch.full((1, 2, 2, 2), 2.0))
        collector.add_pair(2, torch.full((1, 2, 2, 2), 3.0), torch.full((1, 2, 2, 2), 4.0))
        module._save_sharded_pairs(
            output_dir=tmp,
            rank=1,
            shard_id=0,
            collector=collector,
            global_indices=[4],
            seeds=[5046],
            prompts=["prompt 4"],
            total_samples=10,
            world_size=3,
            settings=settings,
        )
        pth = Path(tmp) / "data_output_pairs_rank1_shard00000.pth"
        meta = Path(tmp) / "data_output_pairs_rank1_shard00000.json"
        assert pth.exists()
        assert meta.exists()
        metadata = json.loads(meta.read_text())
        assert metadata["complete"] is True
        assert metadata["global_indices"] == [4]
        assert metadata["seeds"] == [5046]
        assert metadata["world_size"] == 3
        assert metadata["settings"] == settings

        other_collector = module.OutputCollector()
        other_collector.add_pair(1, torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 2))
        module._save_sharded_pairs(
            output_dir=tmp,
            rank=0,
            shard_id=0,
            collector=other_collector,
            global_indices=[1],
            seeds=[5043],
            prompts=["prompt 1"],
            total_samples=10,
            world_size=2,
            settings=settings,
        )

        prompts = [f"prompt {idx}" for idx in range(10)]
        assert module._load_completed_sharded_indices(tmp, rank=1) == {1, 4}
        assert module._load_completed_sharded_indices(
            tmp, rank=1, prompts=prompts, noise_seed_start=5042, total_samples=10, expected_settings=settings
        ) == {1, 4}
        try:
            module._load_completed_sharded_indices(
                tmp, rank=1, prompts=prompts, noise_seed_start=5042, total_samples=11, expected_settings=settings
            )
            raise AssertionError("expected total-sample mismatch")
        except ValueError as exc:
            assert "total-sample metadata" in str(exc)
        bad_settings = dict(settings)
        bad_settings["guidance_scale"] = 7.0
        try:
            module._load_completed_sharded_indices(
                tmp, rank=1, prompts=prompts, noise_seed_start=5042, total_samples=10, expected_settings=bad_settings
            )
            raise AssertionError("expected settings mismatch")
        except ValueError as exc:
            assert "settings metadata" in str(exc)
        prompts[4] = "different prompt"
        try:
            module._load_completed_sharded_indices(
                tmp, rank=1, prompts=prompts, noise_seed_start=5042, total_samples=10, expected_settings=settings
            )
            raise AssertionError("expected prompt mismatch")
        except ValueError as exc:
            assert "prompt metadata" in str(exc)
        prompts[4] = "prompt 4"
        try:
            module._load_completed_sharded_indices(
                tmp, rank=1, prompts=prompts, noise_seed_start=7, total_samples=10, expected_settings=settings
            )
            raise AssertionError("expected seed mismatch")
        except ValueError as exc:
            assert "seed metadata" in str(exc)
        assert module._load_completed_sharded_indices(tmp, rank=0) == {1, 4}
        data = torch.load(pth, map_location="cpu")
        assert data["num_samples"] == 1
        assert data["num_samples_total"] == 10
        assert data["global_indices"] == [4]
        assert data["seeds"] == [5046]
        assert not list(Path(tmp).glob("data_output_pairs_rank1_shard*.tmp*"))
        assert data["fp16_output"][1].shape == (1, 1, 2, 2, 2)
        assert data["quant_output"][2].shape == (1, 1, 2, 2, 2)

    with tempfile.TemporaryDirectory() as tmp:
        orphan_pth = Path(tmp) / "data_output_pairs_rank1_shard00002.pth"
        torch.save({"orphan": True}, orphan_pth)
        incomplete_pth = Path(tmp) / "data_output_pairs_rank1_shard00003.pth"
        incomplete_json = Path(tmp) / "data_output_pairs_rank1_shard00003.json"
        torch.save({"incomplete": True}, incomplete_pth)
        incomplete_json.write_text(json.dumps({"complete": False}), encoding="utf-8")
        other_rank_pth = Path(tmp) / "data_output_pairs_rank0_shard00000.pth"
        torch.save({"other": True}, other_rank_pth)

        assert module._load_completed_sharded_indices(tmp, rank=1, total_samples=10) == set()
        assert not orphan_pth.exists()
        assert not incomplete_pth.exists()
        assert not incomplete_json.exists()
        assert other_rank_pth.exists()


class _DummyConfig:
    in_channels = 2
    out_channels = 2
    timestep_scale = 1.0
    sample_size = 2


class _DummyTransformer:
    def __init__(self, offset: float):
        self.config = _DummyConfig()
        self.dtype = torch.float32
        self._param = torch.nn.Parameter(torch.zeros(()))
        self.offset = offset

    def parameters(self):
        yield self._param

    def __call__(self, latent_model_input, *args, return_dict=False, **kwargs):
        return (latent_model_input.to(torch.float32) + self.offset,)


class _DummyScheduler:
    def set_timesteps(self, steps, device):
        self.timesteps = torch.arange(steps, 0, -1, device=device, dtype=torch.float32)

    def scale_model_input(self, latent_model_input, t):
        return latent_model_input

    def step(self, model_output, t, latents, **kwargs):
        return (latents,)


class _DummyPixArtPipeline:
    def encode_prompt(self, *, prompt, negative_prompt, device, **kwargs):
        batch_size = len(prompt)
        prompt_embeds = torch.ones(batch_size, 1, 2, device=device)
        neg_embeds = -prompt_embeds
        mask = torch.ones(batch_size, 1, device=device)
        return prompt_embeds, mask, neg_embeds, mask.clone()

    def prepare_latents(self, *, batch_size, num_channels_latents, height, width, dtype, device, generator, latents):
        generators = generator if isinstance(generator, list) else [generator]
        assert len(generators) == batch_size
        samples = [torch.randn(num_channels_latents, 2, 2, generator=g, device=device, dtype=dtype) for g in generators]
        return torch.stack(samples, dim=0)

    def prepare_extra_step_kwargs(self, generator, eta):
        return {}


class _DummySanaPipeline:
    def encode_prompt(self, *, prompt, negative_prompt, device, **kwargs):
        batch_size = len(prompt)
        prompt_embeds = torch.ones(batch_size, 1, 2, device=device)
        neg_embeds = -prompt_embeds
        mask = torch.ones(batch_size, 1, device=device)
        return prompt_embeds, mask, neg_embeds, mask.clone()

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        generators = generator if isinstance(generator, list) else [generator]
        assert len(generators) == batch_size
        samples = [torch.randn(num_channels_latents, 2, 2, generator=g, device=device, dtype=dtype) for g in generators]
        return torch.stack(samples, dim=0)

    def prepare_extra_step_kwargs(self, generator, eta):
        return {}


def test_pixart_sharded_helpers_write_metadata_and_resume_indices():
    module = _load_module(PIXART_PATH, "pixart_collect_statistics_under_test")
    _exercise_module(module)


def test_sana_sharded_helpers_write_metadata_and_resume_indices():
    module = _load_module(SANA_PATH, "sana_collect_statistics_under_test")
    _exercise_module(module)


def test_pixart_manual_loop_runs_real_batch_forward_and_legacy_single():
    module = _load_module(PIXART_PATH, "pixart_collect_statistics_batch_under_test")
    pairs = module.manual_denoising_loop(
        transformer_fp16=_DummyTransformer(offset=0.0),
        transformer_quant=_DummyTransformer(offset=10.0),
        pipeline=_DummyPixArtPipeline(),
        scheduler=_DummyScheduler(),
        prompt=["a", "b"],
        negative_prompt="",
        seed=[11, 12],
        num_inference_steps=2,
        guidance_scale=4.5,
        device=torch.device("cpu"),
        height=16,
        width=16,
    )
    assert len(pairs) == 2
    for fp16_out, quant_out in pairs.values():
        assert fp16_out.shape == (2, 2, 2, 2)
        assert quant_out.shape == (2, 2, 2, 2)

    single = module.manual_denoising_loop(
        transformer_fp16=_DummyTransformer(offset=0.0),
        transformer_quant=_DummyTransformer(offset=10.0),
        pipeline=_DummyPixArtPipeline(),
        scheduler=_DummyScheduler(),
        prompt="a",
        negative_prompt="",
        seed=11,
        num_inference_steps=1,
        guidance_scale=1.0,
        device=torch.device("cpu"),
        height=16,
        width=16,
    )
    only_fp16, only_quant = next(iter(single.values()))
    assert only_fp16.shape == (1, 2, 2, 2)
    assert only_quant.shape == (1, 2, 2, 2)


def test_sana_manual_loop_runs_real_batch_forward_and_legacy_single():
    module = _load_module(SANA_PATH, "sana_collect_statistics_batch_under_test")
    pairs = module.manual_denoising_loop(
        pipe=_DummySanaPipeline(),
        transformer_fp16=_DummyTransformer(offset=0.0),
        transformer_quant=_DummyTransformer(offset=10.0),
        scheduler=_DummyScheduler(),
        prompt=["a", "b"],
        seed=[11, 12],
        num_inference_steps=2,
        guidance_scale=4.5,
        device=torch.device("cpu"),
        height=16,
        width=16,
        negative_prompt="",
    )
    assert len(pairs) == 2
    for fp16_out, quant_out in pairs.values():
        assert fp16_out.shape == (2, 2, 2, 2)
        assert quant_out.shape == (2, 2, 2, 2)

    single = module.manual_denoising_loop(
        pipe=_DummySanaPipeline(),
        transformer_fp16=_DummyTransformer(offset=0.0),
        transformer_quant=_DummyTransformer(offset=10.0),
        scheduler=_DummyScheduler(),
        prompt="a",
        seed=11,
        num_inference_steps=1,
        guidance_scale=1.0,
        device=torch.device("cpu"),
        height=16,
        width=16,
        negative_prompt="",
    )
    only_fp16, only_quant = next(iter(single.values()))
    assert only_fp16.shape == (1, 2, 2, 2)
    assert only_quant.shape == (1, 2, 2, 2)
