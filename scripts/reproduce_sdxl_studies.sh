#!/usr/bin/env bash
# SDXL W3A4 study rows: K=10 calibration subsets, ablations, and prior sampler-side corrections.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
TARGET="${1:-help}"

QD="$REPO_ROOT/experiments/svdquant"
PYTHON_BIN="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
BASE_MODEL="${BASE_MODEL:-stabilityai/stable-diffusion-xl-base-1.0}"
QUANT_MODEL_DIR="$(realpath -m "${QUANT_MODEL_DIR:-$QD/sdxl_w3a4/model/unet_w3a4_g64}")"
REF_K=1000

# Downloaded study artifacts. Regeneration targets write to STUDY_OUTPUT_DIR instead.
DOWNLOADED_STUDY_DIR="$QD/sdxl_w3a4_k10_subsets/outputs/calibration_study_pooled_k$REF_K"
DOWNLOADED_K10_DIR="$QD/sdxl_w3a4_k10_subsets/outputs/pooled_scalar_k10"
STUDY_OUTPUT_DIR="$(realpath -m "${STUDY_OUTPUT_DIR:-$DOWNLOADED_STUDY_DIR}")"
EVAL_ROOT="$(realpath -m "${EVAL_ROOT:-$QD/sdxl_w3a4_studies}")"
REGENERATED_PAIRS_DIR="$QD/sdxl_w3a4/outputs/regenerated_calibration"

require_fresh_study_dir() {
  if [ "$STUDY_OUTPUT_DIR" = "$DOWNLOADED_STUDY_DIR" ]; then
    echo "$TARGET writes new statistics; set STUDY_OUTPUT_DIR to a fresh directory." >&2
    exit 1
  fi
}

has_mu_cov() {
  [ -f "$1/mu_dict.npy" ] && [ -f "$1/cov_dict.npy" ]
}

# First REF_K paired FP/quantized outputs of the regenerated calibration pool.
ensure_study_pairs() {
  local out="$STUDY_OUTPUT_DIR/raw_pairs_first$REF_K/data_output_pairs.pth"
  if [ -f "$out" ]; then
    echo "$out"
    return
  fi
  local source=(--data_output_pairs_glob "$REGENERATED_PAIRS_DIR/data_output_pairs_rank*.pth")
  if [ -n "${DATA_OUTPUT_PAIRS:-}" ]; then
    source=(--data_output_pairs_path "$DATA_OUTPUT_PAIRS")
  elif ! compgen -G "$REGENERATED_PAIRS_DIR/data_output_pairs_rank*.pth" >/dev/null; then
    echo "No calibration pairs found. Run: bash scripts/reproduce_sdxl_studies.sh raw-calibration" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$out")"
  "$PYTHON_BIN" scripts/materialize_data_output_pairs_subset.py "${source[@]}" \
    --output_path "$out" --max_samples "$REF_K" --expected_num_samples "$REF_K" >&2
  echo "$out"
}

pooled_reference() {
  local dir="$STUDY_OUTPUT_DIR/reference_pooled_scalar_k$REF_K"
  if ! has_mu_cov "$dir"; then
    "$PYTHON_BIN" scripts/fit_scalar_gaussian.py --data_output_pairs_path "$(ensure_study_pairs)" \
      --output_dir "$dir" --max_samples "$REF_K" --expected_num_samples "$REF_K" >&2
  fi
  echo "$dir"
}

channelwise_reference() {
  local dir="$STUDY_OUTPUT_DIR/reference_channelwise_k$REF_K"
  if ! has_mu_cov "$dir"; then
    "$PYTHON_BIN" "$QD/sdxl_w3a4/scripts/gaussian_modeling.py" \
      --data_output_pairs_path "$(ensure_study_pairs)" --output_dir "$dir" >&2
  fi
  echo "$dir"
}

k10_statistics() {
  local dir
  if [ "$STUDY_OUTPUT_DIR" = "$DOWNLOADED_STUDY_DIR" ]; then
    dir="$DOWNLOADED_K10_DIR/calib_$1"
  else
    dir="$STUDY_OUTPUT_DIR/k10_selection/calib_$1"
  fi
  if ! has_mu_cov "$dir"; then
    echo "Missing K=10 statistics: $dir (run select-k10-trials first)." >&2
    exit 1
  fi
  echo "$dir"
}

COMMON_ARGS=(
  --num_samples "$NUM_SAMPLES" --base_model "$BASE_MODEL" --quant_model_dir "$QUANT_MODEL_DIR"
  --mjhq_prompt_sample_seed 42 --noise_seed_start 42 --num_inference_steps 30 --guidance_scale 7.5
  --no-save_xt
)
[ -n "${META_PATH:-}" ] && COMMON_ARGS+=(--meta_path "$META_PATH")

# SMOKE_SAMPLES=N generates only the first N of the 5,000 prompts and skips metrics.
compute_metrics() {
  local out="$1" key="$2"
  if [ -n "${SMOKE_SAMPLES:-}" ]; then
    echo "SMOKE_SAMPLES set; skipping metrics for $out/images/$key"
    return
  fi
  "$PYTHON_BIN" evaluation/compute_metrics.py --evaluate_dir "$out" --seed 42 \
    --num_samples "$NUM_SAMPLES" --prompts_path "$out/evaluation_results.json" --models "$key"
}

# Q-Drift or D2-DPM on the SDXL W3A4 checkpoint: run_sdxl <name> <stats dir> <image key> <sampler args...>
run_sdxl() {
  local name="$1" stats="$2" key="$3"
  shift 3
  local out="$EVAL_ROOT/$name" range=()
  [ -n "${SMOKE_SAMPLES:-}" ] && range=(--global_indices "0-$((SMOKE_SAMPLES - 1))")
  (cd "$QD/sdxl_w3a4" && torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/evaluate.py \
    "${COMMON_ARGS[@]}" "${range[@]}" --output_dir "$out" \
    --mu_dict "$stats/mu_dict.npy" --cov_dict "$stats/cov_dict.npy" "$@" --skip_fp16 --skip_quant_baseline)
  compute_metrics "$out" "$key"
}

QDRIFT_SCALAR=(--qdrift_scales 1.0 --bias_scales 0.0 --qdrift_scalar)
SCALAR_KEY=quant_bias0.0_qdrift_scale1.0_scalar

case "$TARGET" in
  k10-maxabs)
    run_sdxl "$TARGET" "$(k10_statistics k10_max_sum_abs_delta_c)" "$SCALAR_KEY" "${QDRIFT_SCALAR[@]}" ;;
  k10-min-signed)
    run_sdxl "$TARGET" "$(k10_statistics k10_min_sum_delta_c)" "$SCALAR_KEY" "${QDRIFT_SCALAR[@]}" ;;
  k10-max-signed)
    run_sdxl "$TARGET" "$(k10_statistics k10_max_sum_delta_c)" "$SCALAR_KEY" "${QDRIFT_SCALAR[@]}" ;;
  ablation-scalar)
    run_sdxl "$TARGET" "$(pooled_reference)" "$SCALAR_KEY" "${QDRIFT_SCALAR[@]}" ;;
  ablation-unconditional)
    run_sdxl "$TARGET" "$(pooled_reference)" "${SCALAR_KEY}_uncond" "${QDRIFT_SCALAR[@]}" --qdrift_unconditional ;;
  ablation-channelwise)
    run_sdxl "$TARGET" "$(channelwise_reference)" quant_bias0.0_qdrift_scale1.0 --qdrift_scales 1.0 --bias_scales 0.0 ;;
  prior-d2-deterministic)
    run_sdxl "$TARGET" "$(channelwise_reference)" quant_bias1.0_qdrift_scale0.0 --qdrift_scales 0.0 --bias_scales 1.0 ;;
  prior-d2-stochastic)
    run_sdxl "$TARGET" "$(channelwise_reference)" quant_bias1.0s_qdrift_scale0.0 \
      --qdrift_scales 0.0 --bias_scales 1.0 --bias_stochastic ;;
  prior-ptqd)
    out="$EVAL_ROOT/$TARGET" range=()
    ptqd_stats="$(realpath "${PTQD_STATS:-$QD/sdxl_w3a4_ptqd/calibration/ptqd_stats.npy}")"
    [ -n "${SMOKE_SAMPLES:-}" ] && range=(--sample_start 0 --sample_end "$SMOKE_SAMPLES")
    (cd "$QD/sdxl_w3a4_ptqd" && torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/evaluate.py \
      "${COMMON_ARGS[@]}" "${range[@]}" --output_dir "$out" --ptqd_stats "$ptqd_stats" \
      --ptqd_kt_scales 1.0 --ptqd_bias_scales 1.0 --skip_fp16 --skip_quant_baseline)
    compute_metrics "$out" quant_ptqd_bias1.0_kt1.0 ;;
  prior-qncd)
    out="$EVAL_ROOT/$TARGET" range=()
    [ -n "${SMOKE_SAMPLES:-}" ] && range=(--sample_start 0 --sample_end "$SMOKE_SAMPLES")
    (cd "$QD/sdxl_w3a4_qncd" && torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/evaluate_qncd.py \
      "${COMMON_ARGS[@]}" "${range[@]}" --output_dir "$out" --qncd_mean_scale 1.0 --qncd_reverse_interval 10)
    compute_metrics "$out" quant_qncd_runtime_mean1.0_ri10 ;;
  raw-calibration)
    (cd "$QD/sdxl_w3a4" && STAGE=calibrate CALIB_NUM_SAMPLES="$REF_K" NUM_GPUS="$NUM_GPUS" \
      CALIBRATION_DIR="$REGENERATED_PAIRS_DIR" bash run_paper.sh) ;;
  select-k10-trials)
    require_fresh_study_dir
    pairs="$(ensure_study_pairs)"
    reference="$(pooled_reference)"
    "$PYTHON_BIN" "$QD/sdxl_w3a4_k10_subsets/scripts/sample_subset_trials.py" \
      --data_output_pairs "$pairs" --oracle_cov_dict "$reference/cov_dict.npy" \
      --calib_sizes 10 --num_trials 200 --seed 0 --max_samples "$REF_K" \
      --output_json "$STUDY_OUTPUT_DIR/subset_trials/cfactor_ci_subsample_trials_n200.json"
    "$PYTHON_BIN" "$QD/sdxl_w3a4_k10_subsets/scripts/select_stress_subsets.py" \
      --trials_json "$STUDY_OUTPUT_DIR/subset_trials/cfactor_ci_subsample_trials_n200.json" \
      --data_output_pairs "$pairs" --calib_size 10 --out_dir "$STUDY_OUTPUT_DIR/k10_selection" ;;
  fit-ptqd)
    require_fresh_study_dir
    "$PYTHON_BIN" "$QD/sdxl_w3a4_ptqd/scripts/fit_ptqd_stats.py" \
      --data_output_pairs_path "$(ensure_study_pairs)" --output_dir "$STUDY_OUTPUT_DIR/prior_ptqd_k$REF_K" ;;
  *)
    cat <<EOF
Usage: bash scripts/reproduce_sdxl_studies.sh <target>

Evaluation targets (results under \$EVAL_ROOT/<target>/, default $QD/sdxl_w3a4_studies):
  k10-maxabs k10-min-signed k10-max-signed          K=10 calibration subsets
  ablation-scalar ablation-unconditional ablation-channelwise
  prior-d2-deterministic prior-d2-stochastic prior-ptqd prior-qncd

Regeneration targets (write to a fresh \$STUDY_OUTPUT_DIR):
  raw-calibration     recollect 1,000 paired FP/quantized outputs
  select-k10-trials   redo the 200-trial K=10 subset selection
  fit-ptqd            refit PTQD statistics (then set PTQD_STATS)

Environment: NUM_GPUS, NUM_SAMPLES, META_PATH, BASE_MODEL, QUANT_MODEL_DIR, STUDY_OUTPUT_DIR,
  EVAL_ROOT, PTQD_STATS, DATA_OUTPUT_PAIRS, SMOKE_SAMPLES.
EOF
    ;;
esac
