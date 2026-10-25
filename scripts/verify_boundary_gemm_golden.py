"""Build/evaluate the P12 FP32 Ascend C golden using the canonical boundary."""
import argparse
import json
from pathlib import Path

from kernelbench.backends.boundary import capture_boundary_contract, lower_boundary_contract, render_boundary_scaffold
from kernelbench.backends.spec_factory import AscendCImplementationConfig
from kernelbench.backends.builder import AscendCBuilder
from kernelbench.backends.ascendc import evaluate_ascendc_artifact
from pyramidkernel.core.loader import KernelBenchLoader


def main():
    parser = argparse.ArgumentParser()
    kernelbench_root = Path(__file__).resolve().parents[1]
    parser.add_argument('--kernelbench-root', type=Path, default=kernelbench_root)
    parser.add_argument('--kernel', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', choices=['small','tail','official'], default='small')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--device', type=int, default=3)
    args=parser.parse_args()
    reference_path=args.kernelbench_root/'KernelBench/level2/12_Gemm_Multiply_LeakyReLU.py'
    kernel_path=args.kernel or args.kernelbench_root/'golden/ascendc/level2/12_gemm_multiply_leaky_relu.cpp'
    source=reference_path.read_text()
    if args.profile=='small':
        source += '\ndef get_inputs():\n    return [torch.randn(8, 64)]\ndef get_init_inputs():\n    return [64, 32, 2.0, 0.1]\n'
    elif args.profile=='tail':
        source += '\ndef get_inputs():\n    return [torch.randn(7, 37)]\ndef get_init_inputs():\n    return [37, 19, -1.5, 0.2]\n'
    contract=capture_boundary_contract(source,2,12)
    if contract.status!='supported': raise RuntimeError(contract.failure_message)
    spec=lower_boundary_contract(contract)
    impl=AscendCImplementationConfig(tiling=spec.tiling,block_dim=24,workspace=0)
    spec=lower_boundary_contract(contract,impl)
    info=KernelBenchLoader()._inspect_problem(reference_path)
    candidate=render_boundary_scaffold(info,contract,spec)
    start=candidate.index('# <<<IMPROVE:ascendc_kernel>>>')
    begin=candidate.index('\n',start)+1
    end=candidate.index('# <<<END_IMPROVE>>>',begin)
    candidate=candidate[:begin]+'ASCENDC_KERNEL_SOURCE = '+repr(kernel_path.read_text())+'\n'+candidate[end:]
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'candidate.py').write_text(candidate)
    (args.output/'reference.py').write_text(source)
    (args.output/'contract.json').write_text(contract.canonical_json())
    artifact=AscendCBuilder(args.output/'cache',build_timeout=300,build_jobs=2).build(candidate,level=2,problem_id=12)
    print('BUILD_OK',artifact.root,flush=True)
    if args.build_only:return
    report=evaluate_ascendc_artifact(artifact,source,device_index=args.device,num_correct_trials=3,num_warmup=2,num_perf_trials=3,timeout=300)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report.get(k) for k in ['compiled','correctness','failure_type','custom_op_call_count','speedup','error','correctness_trials']},indent=2),flush=True)


if __name__=='__main__':main()
