#!/usr/bin/env python3
"""Run signed/tail diagnostics using freshly captured contracts and the unified evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from kernelbench.backends.ascendc import evaluate_ascendc_artifact
from kernelbench.backends.boundary import capture_boundary_contract, lower_boundary_contract, render_boundary_scaffold
from kernelbench.backends.builder import AscendCBuilder
from kernelbench.backends.security import validate_candidate_source
from kernelbench.backends.spec import extract_literal_assignment, extract_literal_spec
from kernelbench.backends.spec_factory import AscendCImplementationConfig
from pyramidkernel.core.loader import KernelBenchLoader


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def cases(problem_id):
    if problem_id == 47:
        return [((2, 7, 13), 1), ((1, 9, 17), 1)]
    slopes = (None, 0.2) if problem_id == 20 else (None,)
    return [(shape, slope) for shape in ((8, 64), (1, 1), (3, 17), (2, 8193)) for slope in slopes]


def run_case(args, shape, attribute):
    reference_path = next((args.kernelbench_root / 'KernelBench' / 'level1').glob(f'{args.problem_id}_*.py'))
    original = reference_path.read_text()
    source = original if shape is None else original + f'\n\ndef get_inputs():\n    return [torch.randn({shape!r})]\n'
    if attribute is not None:
        source += f'\ndef get_init_inputs():\n    return [{attribute!r}]\n'
    name = 'official' if shape is None else 'x'.join(map(str, shape)) + ('_attr_' + str(attribute) if attribute is not None else '_default')
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    ref = out / 'reference.py'
    ref.write_text(source)
    golden_path = next((args.kernelbench_root / 'golden' / 'ascendc' / 'level1').glob(f'{args.problem_id}_*.py'))
    golden = golden_path.read_text()
    spec = extract_literal_spec(golden)
    contract = capture_boundary_contract(source, 1, args.problem_id, source_path=str(ref))
    if contract.status != 'supported':
        raise RuntimeError(f'{contract.failure_type}: {contract.failure_message}')
    canonical = lower_boundary_contract(contract, AscendCImplementationConfig.from_spec(spec))
    scaffold = render_boundary_scaffold(KernelBenchLoader()._inspect_problem(ref), contract, canonical)
    kernel = str(extract_literal_assignment(golden, 'ASCENDC_KERNEL_SOURCE'))
    kernel, count = re.subn(rf'(extern\s+"C"\s+__global__\s+__aicore__\s+void\s+){re.escape(spec.entrypoint)}(\s*\()', rf'\g<1>{canonical.entrypoint}\g<2>', kernel)
    if count != 1:
        raise RuntimeError('golden entrypoint was not unique')
    begin = scaffold.index('# <<<IMPROVE:ascendc_kernel>>>')
    end = scaffold.index('# <<<END_IMPROVE>>>', begin)
    body_start = scaffold.index('\n', begin) + 1
    candidate = scaffold[:body_start] + 'ASCENDC_KERNEL_SOURCE = ' + repr(kernel) + '\n' + scaffold[end:]
    (out / 'candidate.py').write_text(candidate)
    (out / 'contract.json').write_text(contract.canonical_json() + '\n')
    started = time.monotonic()
    row = {'problem_id': args.problem_id, 'shape': shape, 'attribute': attribute, 'official_input': shape is None, 'reference_source_hash': sha(original), 'diagnostic_source_hash': sha(source), 'candidate_hash': sha(candidate), 'boundary_contract_hash': contract.contract_hash(), 'spec_hash': sha(canonical.to_json())}
    try:
        validate_candidate_source(candidate)
        artifact = AscendCBuilder(args.output_dir / 'cache', build_jobs=2, build_timeout=300).build(candidate, level=1, problem_id=args.problem_id)
        row['artifact_root'] = str(artifact.root)
        row['result'] = evaluate_ascendc_artifact(artifact, source, device_index=args.device, seed=42, num_correct_trials=3, num_warmup=2, num_perf_trials=1, timeout=300 if shape is None else 120)
    except Exception as exc:
        row['result'] = {'correctness': False, 'failure_type': getattr(exc, 'failure_type', type(exc).__name__), 'error': str(exc)[-4000:]}
    row['elapsed_seconds'] = round(time.monotonic() - started, 3)
    (out / 'report.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps({'case': name, 'problem_id': args.problem_id, 'correct': row['result'].get('correctness'), 'failure_type': row['result'].get('failure_type'), 'seconds': row['elapsed_seconds']}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernelbench-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--problem-id', type=int, choices=(20, 25, 47), required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--tail-only', action='store_true')
    parser.add_argument('--official', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = cases(args.problem_id)
    if args.official:
        selected = [(None, None)]
    if args.tail_only:
        selected = [((3, 17), None)] if args.problem_id != 47 else selected
    rows = []
    for shape, attribute in selected:
        rows.append(run_case(args, shape, attribute))
        report = {'problem_id': args.problem_id, 'cases': rows, 'all_correct': all(r['result'].get('correctness') for r in rows), 'unverified_reduction_axes': [0, 2] if args.problem_id == 47 else []}
        (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if all(r['result'].get('correctness') for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
