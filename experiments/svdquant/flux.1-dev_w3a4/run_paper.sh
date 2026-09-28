#!/usr/bin/env bash
# Paper reproduction wrapper for FLUX.1-dev SVDQuant W3A4.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -n "${QDRIFT_CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$QDRIFT_CONDA_ENV"
fi

STAGE="${STAGE:-all}"
NUM_GPUS="${NUM_GPUS:-6}"
USER_SET_NUM_SAMPLES="${NUM_SAMPLES+x}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
CALIB_NUM_SAMPLES="${CALIB_NUM_SAMPLES:-1000}"
CALIBRATION_SHARD_SIZE="${CALIBRATION_SHARD_SIZE:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_paper}"
CALIBRATION_DIR="${CALIBRATION_DIR:-calibration_pooled_1000}"
BASE_MODEL="${BASE_MODEL:-black-forest-labs/FLUX.1-dev}"
QUANT_MODEL_DIR="${QUANT_MODEL_DIR:-model/transformer_w3a4_g64}"
META_PATH="${META_PATH:-}"
GLOBAL_INDICES="${GLOBAL_INDICES:-${global_indices:-}}"
QDRIFT_ONLY="${QDRIFT_ONLY:-0}"
ALLOW_DOWNLOADS="${ALLOW_DOWNLOADS:-1}"
QDRIFT_SCALES="${QDRIFT_SCALES:-1.0}"
BIAS_SCALES="${BIAS_SCALES:-0.0}"
CALIBRATION_PROMPT_SAMPLING_SEED="${CALIBRATION_PROMPT_SAMPLING_SEED:-5042}"
CALIBRATION_INITIAL_NOISE_SEED_START="${CALIBRATION_INITIAL_NOISE_SEED_START:-5042}"
EVALUATION_PROMPT_SAMPLING_SEED="${EVALUATION_PROMPT_SAMPLING_SEED:-42}"
EVALUATION_INITIAL_NOISE_SEED_START="${EVALUATION_INITIAL_NOISE_SEED_START:-42}"

usage() {
  cat <<EOF
Usage: bash run_paper.sh [options]

Stages:
  STAGE=prepare|quantize   run the original SVDQuant/DeepCompressor quantization command
  STAGE=calibrate          collect paired FP/quant outputs and fit Gaussian stats
  STAGE=generate           generate FP, quantized baseline, and scalar Q-Drift images in one directory
  STAGE=metrics            compute FID/CLIP/PSNR/LPIPS/SSIM from OUTPUT_DIR
  STAGE=all                quantize if needed, then generate and metrics using provided calibration stats
  STAGE=full               quantize, recalibrate, generate, and metrics
  STAGE=smoke              generate a short smoke run (defaults to NUM_SAMPLES=10 if unset)

Common options can be passed as environment variables or CLI flags:
  NUM_GPUS, NUM_SAMPLES, CALIB_NUM_SAMPLES, OUTPUT_DIR, CALIBRATION_DIR,
  BASE_MODEL, QUANT_MODEL_DIR, META_PATH, ALLOW_DOWNLOADS,
  QDRIFT_SCALES, BIAS_SCALES, CALIBRATION_SHARD_SIZE, QDRIFT_ONLY.

GLOBAL_INDICES/global_indices may be used for smoke validation. The script
keeps the full NUM_SAMPLES prompt draw, then generates only the selected global
indices so filenames, prompt IDs, and deterministic noise seeds match the paper
evaluation. Leave it unset for the full paper run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    --calib_num_samples|--calibration_num_samples) CALIB_NUM_SAMPLES="$2"; shift 2 ;;
    --calibration_shard_size|--shard_size) CALIBRATION_SHARD_SIZE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --calibration_dir) CALIBRATION_DIR="$2"; shift 2 ;;
    --base_model) BASE_MODEL="$2"; shift 2 ;;
    --quant_model_dir) QUANT_MODEL_DIR="$2"; shift 2 ;;
    --meta_path) META_PATH="$2"; shift 2 ;;
    --global_indices|--GLOBAL_INDICES) GLOBAL_INDICES="$2"; shift 2 ;;
    --qdrift_only) QDRIFT_ONLY=1; shift ;;
    --allow_downloads) ALLOW_DOWNLOADS=1; shift ;;
    --local_files_only) ALLOW_DOWNLOADS=0; shift ;;
    --qdrift_scales) QDRIFT_SCALES="$2"; shift 2 ;;
    --bias_scales) BIAS_SCALES="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/SVDQuant:$REPO_ROOT/experiments/svdquant:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ "$ALLOW_DOWNLOADS" = "1" ] || [ "$ALLOW_DOWNLOADS" = "true" ]; then
  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 DIFFUSERS_OFFLINE=0
  DOWNLOAD_FLAG="--allow_downloads"
else
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
  DOWNLOAD_FLAG="--local_files_only"
fi

META_ARGS=()
if [ -n "$META_PATH" ]; then
  META_ARGS+=(--meta_path "$META_PATH")
fi
HEIGHT_WIDTH_ARGS=()
HEIGHT_WIDTH_ARGS+=(--height 1024 --width 1024)
EXTRA_ARGS=()
EVAL_NOISE_ARGS=()
if [ -n "$EVALUATION_INITIAL_NOISE_SEED_START" ]; then
  EVAL_NOISE_ARGS+=(--noise_seed_start "$EVALUATION_INITIAL_NOISE_SEED_START")
fi
GLOBAL_INDEX_ARGS=()
if [ -n "$GLOBAL_INDICES" ]; then
  GLOBAL_INDEX_ARGS+=(--global_indices "$GLOBAL_INDICES")
fi
QDRIFT_ONLY_ARGS=()
if [ "$QDRIFT_ONLY" = "1" ] || [ "$QDRIFT_ONLY" = "true" ]; then
  QDRIFT_ONLY_ARGS+=(--qdrift_only)
fi

verify_quant_artifacts() {
  if [ -f "$QUANT_MODEL_DIR/model.pt" ]; then
    return 0
  fi
  if [ -d "$QUANT_MODEL_DIR/transformer" ]; then
    return 0
  fi
  echo "Quantization did not produce the expected DeepCompressor checkpoint in $QUANT_MODEL_DIR (missing model.pt)." >&2
  exit 1
}

run_quantize() {
  echo "[run_paper] quantize: scripts/quantize_flux_1_dev_w3a4.py -> $QUANT_MODEL_DIR"
  if [ -f "$QUANT_MODEL_DIR/model.pt" ] || [ -d "$QUANT_MODEL_DIR/transformer" ]; then
    echo "[run_paper] quantize: found existing artifacts in $QUANT_MODEL_DIR; skipping."
    verify_quant_artifacts
    return 0
  fi
  python -u scripts/quantize_flux_1_dev_w3a4.py \
    --base_model "$BASE_MODEL" \
    --output_root "$(dirname "$QUANT_MODEL_DIR")" --output_name "$(basename "$QUANT_MODEL_DIR")" --group_size 64 --backend deepcompressor --quant_config flux_w3a4.yaml \
    "$DOWNLOAD_FLAG"
  verify_quant_artifacts
}

run_calibrate() {
  echo "[run_paper] calibrate: writing $CALIBRATION_DIR from $CALIB_NUM_SAMPLES samples"
  if [ -f "$CALIBRATION_DIR/mu_dict.npy" ] || [ -f "$CALIBRATION_DIR/cov_dict.npy" ]; then
    echo "Refusing to recalibrate into $CALIBRATION_DIR because calibration statistics already exist. Set CALIBRATION_DIR to a fresh directory to preserve provided artifacts." >&2
    exit 1
  fi
  if [ -z "$META_PATH" ]; then
    echo "STAGE=calibrate requires META_PATH/--meta_path pointing to MJHQ meta_data.json so calibration prompts can exclude evaluation IDs." >&2
    exit 2
  fi
  mkdir -p "$CALIBRATION_DIR"
  SPLIT_DIR="$CALIBRATION_DIR/prompt_split"
  python "$REPO_ROOT/scripts/prepare_calibration_split.py" \
    --meta_path "$META_PATH" \
    --output_dir "$SPLIT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --calibration_num_samples "$CALIB_NUM_SAMPLES" \
    --evaluation_seed "$EVALUATION_PROMPT_SAMPLING_SEED" \
    --calibration_seed "$CALIBRATION_PROMPT_SAMPLING_SEED"
  CALIBRATION_PROMPT_FILE="$SPLIT_DIR/eligible_calibration_metadata.json"
  torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/collect_statistics.py \
    --output_dir "$CALIBRATION_DIR" \
    --use_mjhq \
    --num_samples "$CALIB_NUM_SAMPLES" \
    --shard_size "$CALIBRATION_SHARD_SIZE" \
    --base_model "$BASE_MODEL" \
    --quant_model_dir "$QUANT_MODEL_DIR" \
    --noise_seed_start "$CALIBRATION_INITIAL_NOISE_SEED_START" \
    --mjhq_prompt_sample_seed "$CALIBRATION_PROMPT_SAMPLING_SEED" \
    --num_inference_steps 20 \
    --guidance_scale 3.5 \
    --height 1024 --width 1024 \
    --prompt_file "$CALIBRATION_PROMPT_FILE"
  python "$REPO_ROOT/scripts/fit_scalar_gaussian.py" \
    --data_output_pairs_glob "$CALIBRATION_DIR/data_output_pairs_rank*_shard*.pth" \
    --output_dir "$CALIBRATION_DIR" \
    --expected_num_samples "$CALIB_NUM_SAMPLES" \
    --max_samples "$CALIB_NUM_SAMPLES"
}
run_generate() {
  if [ ! -f "$CALIBRATION_DIR/mu_dict.npy" ] || [ ! -f "$CALIBRATION_DIR/cov_dict.npy" ]; then
    echo "Missing $CALIBRATION_DIR/mu_dict.npy or cov_dict.npy. Run STAGE=calibrate or keep the provided calibration files." >&2
    exit 1
  fi
  echo "[run_paper] generate: FP, quantized baseline, and scalar Q-Drift into $OUTPUT_DIR"
  torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/evaluate.py \
    --mu_dict "$CALIBRATION_DIR/mu_dict.npy" \
    --cov_dict "$CALIBRATION_DIR/cov_dict.npy" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --mjhq_prompt_sample_seed "$EVALUATION_PROMPT_SAMPLING_SEED" \
    "${EVAL_NOISE_ARGS[@]}" \
    --num_inference_steps 20 \
    --guidance_scale 3.5 \
    --base_model "$BASE_MODEL" \
    --quant_model_dir "$QUANT_MODEL_DIR" \
    "${HEIGHT_WIDTH_ARGS[@]}" \
    "${META_ARGS[@]}" \
    "${GLOBAL_INDEX_ARGS[@]}" \
    "${QDRIFT_ONLY_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --bias_scales $BIAS_SCALES \
    --qdrift_scales $QDRIFT_SCALES \
    --qdrift_scalar \
    --no-save_xt
}

run_metrics() {
  echo "[run_paper] metrics: $OUTPUT_DIR"
  python "$REPO_ROOT/evaluation/compute_metrics.py" \
    --evaluate_dir "$OUTPUT_DIR" \
    --seed "$EVALUATION_PROMPT_SAMPLING_SEED" \
    --num_samples "$NUM_SAMPLES" \
    --prompts_path "$OUTPUT_DIR/evaluation_results.json"
  python "$REPO_ROOT/evaluation/validate_metrics.py" \
    "$OUTPUT_DIR/evaluation_metrics_only.json" --num-samples "$NUM_SAMPLES"
}

case "$STAGE" in
  prepare|quantize) run_quantize ;;
  calibrate) run_calibrate ;;
  generate) run_generate ;;
  metrics) run_metrics ;;
  all) run_quantize; run_generate; run_metrics ;;
  full) run_quantize; run_calibrate; run_generate; run_metrics ;;
  smoke)
    if [ -z "${USER_SET_NUM_SAMPLES:-}" ]; then
      NUM_SAMPLES=10
    fi
    if [ "$NUM_SAMPLES" -lt 10 ]; then
      echo "STAGE=smoke requires NUM_SAMPLES>=10 because MJHQ stratified sampling can yield empty subsets below 10." >&2
      exit 2
    fi
    run_generate
    ;;
  *) echo "Unknown STAGE: $STAGE" >&2; usage >&2; exit 2 ;;
esac
