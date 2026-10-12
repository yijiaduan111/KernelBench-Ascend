# Ascend NPU Remediation Results

The first Level 1-3 audit reported 241 supported and 9 excluded problems.
The exclusions were rechecked without changing the official problem source.

## Restored problems

- L1/P45: the original audit cloned a 17.2 GiB input before the repeat check.
  The memory-safe worker skips that extra clone for inputs above 8 GiB. The
  reference now runs successfully in 4.56 s on an Ascend 910B3.
- L3/P48 and L3/P49: installing `einops` restored both Mamba references.

## Remaining exclusions

- L1/P41 and L1/P43: the CANN 9.0.1 MaxPool kernels reject the official
  dilation values (3). These need a semantic NPU reference adapter or an
  Ascend C reference implementation; the official source must not be changed.
- L3/P39 through L3/P42: after installing `decorator`, `scipy`, `attrs`, and
  `psutil`, CANN reaches `DynamicGRUV2` and rejects the official FP32 weights.
  The available kernel contract in this environment supports FP16 weights in
  FRACTAL_Z format. These problems therefore remain unsupported for the
  unmodified FP32 suite.

## Recommended suites

The corrected main NPU suite contains 244 problems. The six remaining cases
stay in an excluded/diagnostic suite until an explicit reference adapter is
implemented and validated against the original semantics.
