# Multigroup Random-Ray Transport Benchmark Kernels

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Standalone C++ examples accompanying the unpublished manuscript:

> **Loop reordering and lock-based synchronization in multigroup random-ray transport**
>
> Shihong Li, Shuai Qin, Xiangchun Tian, Jiacheng Li, Qian Zhang
>
> School of Physics, Zhejiang University

These examples demonstrate local flux attenuation and regional accumulation
using synthetic inputs. They are not the THETA solver or a reproduction
package for the PCA-Replica, Iron-88, and Savannah results. Those results use
the full transport code, model inputs, cross-section data, and archived run
settings described in the manuscript. Timings from these examples are not
the paper's transport times or speedups.

The original [`v1.0.0`](https://github.com/poet7/random-ray-loop-reorder-and-locks/tree/v1.0.0)
snapshot is retained unchanged. Use the current `main` branch for the matched
loop-order control, input validation, and threshold-based checks below.

## Local flux attenuation

Each kernel evaluates the directional source from angular-moment coefficients
and uses it to advance the angular flux across one segment. The coefficient
layout is `[moment][group]`, with energy groups contiguous.
All variants use `-expm1(-tau)` to evaluate the attenuation factor
`1 - exp(-tau)` accurately for small optical thicknesses.

| Variant | Source-summation order | Explicit SIMD in source sum | Temporary source vector | Flux update |
| --- | --- | --- | --- | --- |
| (a) `attenuate_group_major` | Group, then moment | No | No | Immediately after each group sum |
| (b) `attenuate_moment_major` | Moment, then group | Yes | Yes | Separate group loop |
| (c) `attenuate_group_major_separate_update` | Group, then moment | No | Yes | Separate group loop |
| (d) `attenuate_moment_major_control` | Moment, then group | No | Yes | Separate group loop |

The program reports two different comparisons:

- **Complete implementations, (a) versus (b):** illustrates the organization
  used in the PCA-Replica comparison, including explicit SIMD, temporary
  storage, and a separate flux update.
- **Controlled loop order, (c) versus (d):** both source sums omit explicit
  SIMD directives, both use the same aligned temporary storage, and both
  call the same flux-update function with an explicit SIMD directive. Only
  the source-summation loop nesting differs, matching the control used for
  Iron-88. The compiler may still automatically vectorize either source sum.

Local attenuation timings are single-threaded. `--threads` sets the worker
count for the regional accumulation tests, not the local projection tests.

## Regional accumulation

Workers add segment contributions to shared regional angular-moment tallies.
The protected accumulation is the same for three lock implementations:

- **Native OpenMP lock:** uses `omp_lock_t` with no additional padding.
  Its waiting behavior depends on the OpenMP runtime and its settings.
- **Conventional spinlock:** uses POSIX `pthread_spinlock_t` in a
  64-byte-aligned object.
- **MCS queue lock:** uses a 64-byte-aligned lock object. Each worker waits
  on its own queue node and receives ownership from its predecessor.

The synthetic tests use 141 and 1410 target regions with uniformly sampled
requests. They do not trace rays through a geometry or reproduce the
spatial and temporal request patterns in the Savannah calculations.
Results compare these specific lock implementations, including their
different object layouts.

## Build and test

Requires Linux (including WSL), a C++17 compiler such as GCC 9.4 or newer,
CMake 3.14 or newer, OpenMP, and POSIX threads. Native MSVC builds are not
supported because the spinlock uses the POSIX API.

```bash
git clone https://github.com/poet7/random-ray-loop-reorder-and-locks.git
cd random-ray-loop-reorder-and-locks
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build
ctest --output-on-failure
```

From the build directory, run a short numerical check or the full synthetic
benchmark:

```bash
./rr_benchmark --smoke --threads 2
./rr_benchmark --threads 8
```

The default full run uses five million local trials and two million regional
requests per mesh. It performs one timing of each variant; it does not
estimate confidence intervals. Smoke mode uses 1024 local trials, 4096
regional requests per mesh, and at most four workers unless explicitly set.
Smoke timings are for functional checks, not performance conclusions.

### Command-line options

| Option | Meaning | Default |
| --- | --- | --- |
| `--smoke` | Use the short functional-check workload | Off |
| `--threads N` | Positive number of OpenMP workers | OpenMP maximum, or at most four in smoke mode |
| `--groups G` | Energy groups; `1 <= G <= 256` | 47 |
| `--moments M` | Positive number of angular moments | 16 |
| `--segments S` | Positive request count per regional test | 2000000, or 4096 in smoke mode |
| `--trials T` | Positive trial count per local kernel | 5000000, or 1024 in smoke mode |
| `--help` | Print usage | |

Explicit workload options override smoke defaults regardless of their order.
The fixed-size source workspace supports at most 256 groups. Invalid,
out-of-range, missing, or overflowing integer arguments are rejected before
kernel execution. The attenuation entry points also check group and moment
counts, including when called directly.

## Numerical verification

Both outgoing angular fluxes and angular-flux changes are compared between
the local variants. Regional tallies are compared with one another and with
an independent reference formed from the number of requests to each region
and the known per-request contribution.

Every comparison must satisfy both a relative L2 difference and a
reference-maximum-normalized absolute difference of at most `1e-8`.
Nonfinite values fail. An all-zero reference passes only when the test is
also zero. A failed comparison produces `[FAIL]` and a nonzero exit status;
`ALL CHECKS PASSED` is printed only if every comparison passes.

CTest covers the attenuation formulas for 1, 47, and 256 groups with
1, 4, and 16 moments, zero and short segments, invalid command-line inputs,
and the error checker itself, including out-of-tolerance, zero-reference,
NaN, and infinity cases. Checks run outside the benchmark's timed intervals.

## Manuscript citation

The manuscript is unpublished. No journal or DOI is assigned here.

```bibtex
@unpublished{li2026loopreordering,
  title={Loop reordering and lock-based synchronization in multigroup random-ray transport},
  author={Li, Shihong and Qin, Shuai and Tian, Xiangchun and Li, Jiacheng and Zhang, Qian},
  note={Unpublished manuscript},
  year={2026}
}
```

## License

[MIT](LICENSE).
