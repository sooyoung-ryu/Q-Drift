#!/bin/bash
# Reproduce the Q-Drift paper row for SDXL-Turbo + MixDQ W4A8.
set -e
set -o pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../../.. && pwd)"

usage() {
    cat <<'EOF'
Usage: bash run_paper.sh [stage ...]

Stages:
  prepare     Create/check mixdq_qdiff_ckpt/config.yaml + ckpt.pth.
  calibrate   Reuse downloaded stats, or regenerate when FORCE_CALIBRATION=1.
  generate    Generate quant_baseline and scalar Q-Drift images. The FP16 reference is the
              SDXL-Turbo FP16 set from experiments/svdquant/sdxl-turbo_w3a4 (run that row first).
  metrics     Compute FID/CLIP/PSNR/LPIPS/SSIM from generated images.
  all         Run prepare calibrate generate metrics.

Default stages: prepare generate metrics

Common environment variables:
  MIXDQ_CKPT=/path/to/ckpt.pth       Existing official MixDQ qdiff checkpoint.
  MIXDQ_BASE_PATH=./mixdq_qdiff_ckpt Directory containing config.yaml and ckpt.pth.
  NUM_GPUS=4                         torchrun processes per node.
  NUM_SAMPLES=5000                   Evaluation sample count.
  GLOBAL_INDICES=0,137,2475          Optional comma-separated global_idx smoke subset.
  GLOBAL_INDICES_SEED_WORLD_SIZE=4   World size whose seed layout is reproduced.
  CALIB_NUM_SAMPLES=1000             Calibration sample count when regenerating stats.
  QDRIFT_ONLY=1                      Generate only Q-Drift images.
  CALIBRATION_DIR=./calibration_pooled_1000 Directory for pooled calibration stats.
  FORCE_CALIBRATION=1                Recompute CALIBRATION_DIR/mu_dict.npy and cov_dict.npy.
  MJHQ_META_PATH=/path/meta_data.json Reuse local MJHQ metadata. Required for fresh calibration
                                      split construction and passed as --meta_path during evaluation.
  CALIBRATION_PROMPT_SAMPLING_SEED=5042 Disjoint calibration prompt split.
  CALIBRATION_INITIAL_NOISE_SEED_START=5042 Calibration initial-noise seed start.
  EVALUATION_PROMPT_SAMPLING_SEED=42 Evaluation prompt/image seed.
  OUT=./evaluation_paper      Output directory for generated images and metrics.
  FP16_REFERENCE_DIR=...             FP16 images (default: SDXL-Turbo SVDQuant evaluation_paper/images/fp16).
  METRICS_RESULTS_PATH=...           Optional JSON metrics output path.
  METRICS_SUMMARY_PATH=...           Optional text metrics output path.
  QDRIFT_CONDA_ENV=qdrift            Optional conda env to activate.
  TORCHRUN_MASTER_PORT=29501         Optional fixed torchrun port (default: --standalone).

Examples:
  MIXDQ_CKPT=/path/to/ckpt.pth NUM_GPUS=4 bash run_paper.sh
  STAGES="prepare generate metrics" NUM_SAMPLES=5000 bash run_paper.sh
  FORCE_CALIBRATION=1 NUM_GPUS=4 bash run_paper.sh calibrate
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 ]]; then
    STAGES="$*"
else
    STAGES=${STAGES:-"prepare generate metrics"}
fi

for stage in $STAGES; do
    case "$stage" in
        prepare|calibrate|generate|metrics|all) ;;
        *) echo "Unknown stage: $stage" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "${QDRIFT_CONDA_ENV:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: QDRIFT_CONDA_ENV=$QDRIFT_CONDA_ENV was requested, but conda is not on PATH." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$QDRIFT_CONDA_ENV"
fi

BASE=${BASE:-$PWD}
NUM_GPUS=${NUM_GPUS:-4}
NUM_SAMPLES=${NUM_SAMPLES:-5000}
CALIB_NUM_SAMPLES=${CALIB_NUM_SAMPLES:-1000}
CALIBRATION_DIR=${CALIBRATION_DIR:-$BASE/calibration_pooled_1000}
CALIBRATION_PROMPT_SAMPLING_SEED=${CALIBRATION_PROMPT_SAMPLING_SEED:-5042}
CALIBRATION_INITIAL_NOISE_SEED_START=${CALIBRATION_INITIAL_NOISE_SEED_START:-5042}
EVALUATION_PROMPT_SAMPLING_SEED=${EVALUATION_PROMPT_SAMPLING_SEED:-42}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-4}
MIXDQ_BASE_PATH=${MIXDQ_BASE_PATH:-$BASE/mixdq_qdiff_ckpt}
MIXDQ_CKPT=${MIXDQ_CKPT:-$MIXDQ_BASE_PATH/ckpt.pth}
OUT=${OUT:-$BASE/evaluation_paper}
CACHE_DIR=${CACHE_DIR:-$BASE/cache}
TORCH_CACHE_DIR=${TORCH_CACHE_DIR:-$CACHE_DIR/torch}
METRICS_CACHE_DIR=${METRICS_CACHE_DIR:-$CACHE_DIR/metrics}
METRICS_RESULTS_PATH=${METRICS_RESULTS_PATH:-$OUT/evaluation_metrics_only.json}
FP16_REFERENCE_DIR=${FP16_REFERENCE_DIR:-$REPO_ROOT/experiments/svdquant/sdxl-turbo_w3a4/evaluation_paper/images/fp16}
METRICS_SUMMARY_PATH=${METRICS_SUMMARY_PATH:-$OUT/evaluation_metrics_only_summary.txt}
EVAL_META_ARGS=()
EVAL_GLOBAL_ARGS=()
QDRIFT_ONLY_ARGS=()
if [[ -n "${MJHQ_META_PATH:-}" ]]; then
    EVAL_META_ARGS=(--meta_path "$MJHQ_META_PATH")
fi
if [[ -n "${GLOBAL_INDICES:-}" ]]; then
    EVAL_GLOBAL_ARGS+=(--global_indices "$GLOBAL_INDICES")
fi
if [[ -n "${GLOBAL_INDICES_SEED_WORLD_SIZE:-4}" ]]; then
    EVAL_GLOBAL_ARGS+=(--global_indices_seed_world_size "${GLOBAL_INDICES_SEED_WORLD_SIZE:-4}")
fi
if [[ "${QDRIFT_ONLY:-0}" == "1" || "${QDRIFT_ONLY:-0}" == "true" ]]; then
    QDRIFT_ONLY_ARGS+=(--qdrift_only)
fi

case "$NUM_GPUS" in ''|*[!0-9]*) echo "ERROR: NUM_GPUS must be a positive integer, got '$NUM_GPUS'." >&2; exit 2 ;; esac
case "$NUM_SAMPLES" in ''|*[!0-9]*) echo "ERROR: NUM_SAMPLES must be a positive integer, got '$NUM_SAMPLES'." >&2; exit 2 ;; esac
case "$CALIB_NUM_SAMPLES" in ''|*[!0-9]*) echo "ERROR: CALIB_NUM_SAMPLES must be a positive integer, got '$CALIB_NUM_SAMPLES'." >&2; exit 2 ;; esac
if (( NUM_GPUS < 1 || NUM_SAMPLES < 1 || CALIB_NUM_SAMPLES < 1 )); then
    echo "ERROR: NUM_GPUS, NUM_SAMPLES, and CALIB_NUM_SAMPLES must be positive." >&2
    exit 2
fi

torchrun_prefix() {
    if [[ -n "${TORCHRUN_MASTER_PORT:-}" ]]; then
        echo --master_port "$TORCHRUN_MASTER_PORT" --nproc_per_node="$NUM_GPUS"
    else
        echo --standalone --nproc_per_node="$NUM_GPUS"
    fi
}

print_config() {
    echo "Q-Drift MixDQ paper reproduction"
    echo "  stages:              $STAGES"
    echo "  base:                $BASE"
    echo "  num_gpus:            $NUM_GPUS"
    echo "  num_samples:         $NUM_SAMPLES"
    echo "  calib_num_samples:   $CALIB_NUM_SAMPLES"
    echo "  calibration_dir:     $CALIBRATION_DIR"
    echo "  calibration_seed:    prompts=$CALIBRATION_PROMPT_SAMPLING_SEED noise=$CALIBRATION_INITIAL_NOISE_SEED_START"
    echo "  evaluation_seed:     $EVALUATION_PROMPT_SAMPLING_SEED"
    echo "  seed_source_world:   ${GLOBAL_INDICES_SEED_WORLD_SIZE:-4}"
    echo "  mixdq_base_path:     $MIXDQ_BASE_PATH"
    echo "  output:              $OUT"
    if [[ -n "${MJHQ_META_PATH:-}" ]]; then
        echo "  mjhq_meta_path:      $MJHQ_META_PATH"
    fi
    if [[ -n "${GLOBAL_INDICES:-}" ]]; then
        echo "  global_indices:      $GLOBAL_INDICES"
        echo "  metrics:             disabled for global-index smoke subset"
    fi
    if [[ "${QDRIFT_ONLY:-0}" == "1" || "${QDRIFT_ONLY:-0}" == "true" ]]; then
        echo "  qdrift_only:         true"
    fi
}

run_prepare() {
    mkdir -p "$MIXDQ_BASE_PATH"
    if [[ ! -f "$MIXDQ_BASE_PATH/config.yaml" ]]; then
        cp mixdq_qdiff_ckpt/config.yaml "$MIXDQ_BASE_PATH/config.yaml"
    fi

    if [[ -f "$MIXDQ_BASE_PATH/ckpt.pth" ]]; then
        echo "prepare: found $MIXDQ_BASE_PATH/ckpt.pth"
    elif [[ -n "${MIXDQ_CKPT:-}" && -f "$MIXDQ_CKPT" ]]; then
        ln -s "$(realpath "$MIXDQ_CKPT")" "$MIXDQ_BASE_PATH/ckpt.pth"
        echo "prepare: linked $MIXDQ_BASE_PATH/ckpt.pth -> $MIXDQ_CKPT"
    else
        echo "prepare: downloading official MixDQ checkpoint"
        python scripts/download_official_mixdq_ckpt.py --out "$MIXDQ_BASE_PATH/ckpt.pth"
    fi

    python scripts/validate_qdiff_ckpt.py         --ckpt "$MIXDQ_BASE_PATH/ckpt.pth"         --config "$MIXDQ_BASE_PATH/config.yaml"
}

run_calibrate() {
    if [[ -z "${MJHQ_META_PATH:-}" ]]; then
        echo "calibrate: MJHQ_META_PATH is required so calibration prompts can exclude evaluation IDs" >&2
        exit 2
    fi
    mkdir -p "$CALIBRATION_DIR"
    if [[ "${FORCE_CALIBRATION:-0}" != "1" && -f "$CALIBRATION_DIR/mu_dict.npy" && -f "$CALIBRATION_DIR/cov_dict.npy" ]]; then
        echo "calibrate: using existing $CALIBRATION_DIR/mu_dict.npy and cov_dict.npy"
        return
    fi
    if [[ "${FORCE_CALIBRATION:-0}" == "1" && ( -f "$CALIBRATION_DIR/mu_dict.npy" || -f "$CALIBRATION_DIR/cov_dict.npy" || -f "$CALIBRATION_DIR/data_output_pairs.pth" ) ]]; then
        echo "calibrate: refusing to overwrite existing files in $CALIBRATION_DIR; set CALIBRATION_DIR to a fresh directory" >&2
        exit 2
    fi

    SPLIT_DIR="$CALIBRATION_DIR/prompt_split"
    python "$REPO_ROOT/scripts/prepare_calibration_split.py"         --meta_path "$MJHQ_META_PATH"         --output_dir "$SPLIT_DIR"         --num_samples "$NUM_SAMPLES"         --calibration_num_samples "$CALIB_NUM_SAMPLES"         --evaluation_seed "$EVALUATION_PROMPT_SAMPLING_SEED"         --calibration_seed "$CALIBRATION_PROMPT_SAMPLING_SEED"
    CALIBRATION_PROMPT_FILE="$SPLIT_DIR/eligible_calibration_metadata.json"

    torchrun $(torchrun_prefix) scripts/collect_statistics.py         --output_dir "$CALIBRATION_DIR"         --use_mjhq         --num_samples "$CALIB_NUM_SAMPLES"         --noise_seed_start "$CALIBRATION_INITIAL_NOISE_SEED_START"         --mjhq_prompt_sample_seed "$CALIBRATION_PROMPT_SAMPLING_SEED"         --num_inference_steps "$NUM_INFERENCE_STEPS"         --guidance_scale 0.0         --quant_backend qdiff         --w_bit 4         --a_bit 8         --mixdq_base_path "$MIXDQ_BASE_PATH"         --prompt_file "$CALIBRATION_PROMPT_FILE"

    python "$REPO_ROOT/scripts/fit_scalar_gaussian.py"         --data_output_pairs_path "$CALIBRATION_DIR/data_output_pairs.pth"         --output_dir "$CALIBRATION_DIR"         --expected_num_samples "$CALIB_NUM_SAMPLES"         --max_samples "$CALIB_NUM_SAMPLES"
}

run_generate() {
    if [[ ! -f "$CALIBRATION_DIR/mu_dict.npy" || ! -f "$CALIBRATION_DIR/cov_dict.npy" ]]; then
        echo "generate: missing calibration stats in $CALIBRATION_DIR; run calibrate first" >&2
        exit 1
    fi
    if [[ ! -f "$MIXDQ_BASE_PATH/config.yaml" || ! -f "$MIXDQ_BASE_PATH/ckpt.pth" ]]; then
        echo "generate: missing MixDQ config/checkpoint; run prepare first" >&2
        exit 1
    fi

    torchrun $(torchrun_prefix) scripts/evaluate.py         --mu_dict "$CALIBRATION_DIR/mu_dict.npy"         --cov_dict "$CALIBRATION_DIR/cov_dict.npy"         --num_samples "$NUM_SAMPLES"         --output_dir "$OUT"         --mjhq_prompt_sample_seed "$EVALUATION_PROMPT_SAMPLING_SEED"         --num_inference_steps "$NUM_INFERENCE_STEPS"         --guidance_scale 0.0         --bias_scales 0.0         --qdrift_scales 1.0         --qdrift_scalar         --quant_backend qdiff         --w_bit 4         --a_bit 8         --mixdq_base_path "$MIXDQ_BASE_PATH"         --fp16_reference_dir "$FP16_REFERENCE_DIR"         --torch_cache_dir "$TORCH_CACHE_DIR"         "${EVAL_META_ARGS[@]}"         "${EVAL_GLOBAL_ARGS[@]}"         "${QDRIFT_ONLY_ARGS[@]}"
}

run_metrics() {
    if [[ -n "${GLOBAL_INDICES:-}" ]]; then
        echo "metrics: skipped because GLOBAL_INDICES is set for a smoke subset"
        return
    fi
    mkdir -p "$OUT/images"
    ln -sfn "$(realpath "$FP16_REFERENCE_DIR")" "$OUT/images/fp16"
    python "$REPO_ROOT/evaluation/compute_metrics.py"         --evaluate_dir "$OUT"         --seed 42         --num_samples "$NUM_SAMPLES"         --prompts_path "$OUT/evaluation_results.json"         --models fp16 quant_baseline quant_bias_0.0_drift1.0_scalar         --cache_dir "$METRICS_CACHE_DIR"         --results_path "$METRICS_RESULTS_PATH"         --summary_path "$METRICS_SUMMARY_PATH"
    python "$REPO_ROOT/evaluation/validate_metrics.py" "$METRICS_RESULTS_PATH" --num-samples "$NUM_SAMPLES"
}

print_config
for stage in $STAGES; do
    case "$stage" in
        prepare) run_prepare ;;
        calibrate) run_calibrate ;;
        generate) run_generate ;;
        metrics) run_metrics ;;
        all) run_prepare; run_calibrate; run_generate; run_metrics ;;
    esac
done

echo "Q-Drift MixDQ paper reproduction finished."
