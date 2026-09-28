"""Prepare stratified MJHQ calibration prompts disjoint from evaluation IDs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def sample_rows(metadata, count, seed):
    groups = defaultdict(list)
    for image_id, info in metadata.items():
        if isinstance(info, dict) and (info.get('prompt') or '').strip():
            groups[str(info['category'])].append({'id': image_id, 'category': str(info['category']), 'prompt': info['prompt'].strip()})
    if len(groups) != 10 or count <= 0 or count % 10:
        raise ValueError('Expected 10 categories and a positive sample count divisible by 10')
    rng = random.Random(seed)
    rows = [row for category in sorted(groups) for row in rng.sample(groups[category], count // 10)]
    rng.shuffle(rows)
    return [dict(row, global_idx=i) for i, row in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--calibration_num_samples', type=int, default=None)
    parser.add_argument('--evaluation_seed', type=int, default=42)
    parser.add_argument('--calibration_seed', type=int, default=5042)
    args = parser.parse_args()
    metadata = json.loads(args.meta_path.read_text())
    evaluation = sample_rows(metadata, args.num_samples, args.evaluation_seed)
    excluded_ids = {row['id'] for row in evaluation}
    eligible = {key: value for key, value in metadata.items() if key not in excluded_ids}
    calibration_count = args.num_samples if args.calibration_num_samples is None else args.calibration_num_samples
    calibration = sample_rows(eligible, calibration_count, args.calibration_seed)
    assert not excluded_ids.intersection(row['id'] for row in calibration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        'eligible_calibration_metadata.json': eligible,
        'calibration_prompts.json': {'num_samples': len(calibration), 'prompt_seed': args.calibration_seed, 'prompts': calibration},
        'evaluation_prompts.json': {'num_samples': len(evaluation), 'prompt_seed': args.evaluation_seed, 'prompts': evaluation},
    }
    for name, payload in outputs.items():
        path = args.output_dir / name
        if path.exists() and json.loads(path.read_text()) != payload:
            raise FileExistsError(f'Refusing to replace a different split: {name}')
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(f'Evaluation: {len(evaluation)}; calibration: {len(calibration)}; overlapping IDs: 0')


if __name__ == '__main__':
    main()
