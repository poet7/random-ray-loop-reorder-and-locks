# Multigroup Random-Ray Transport Optimization Benchmark
### Loop Reordering and Lock-Based Synchronization

[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/poet7/random-ray-loop-reorder-and-locks/releases/tag/v1.0.0)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green.svg)](https://www.openmp.org/)

This repository provides standalone C++ benchmark kernels for the loop-reordering (Algorithm 1) and lock-synchronization (Algorithm 2) methods described in:

> **"Loop reordering and lock-based synchronization in multigroup random-ray transport"**  
> Shihong Li, Shuai Qin, Xiangchun Tian, Jiacheng Li, Qian Zhang*  
> *School of Physics, Zhejiang University*  
> Submitted to *Computer Physics Communications*.

These standalone kernels isolate and verify the core computational algorithms without proprietary reactor geometry or cross-section dependencies. Full-core performance results and scaling analyses in the paper were obtained using the THETA transport code.

The code version corresponding to the manuscript submission is tagged as [`v1.0.0`](https://github.com/poet7/random-ray-loop-reorder-and-locks/releases/tag/v1.0.0).

---

## Core Algorithms

### 1. Loop Reordering in Local Flux Attenuation (Algorithm 1)
Under the flat-source approximation, each ray segment crossing requires evaluating directional sources from moment expansions and updating multigroup angular fluxes:
- **(a) Group-Major Order**: Outer loop over energy groups $g$, performing scalar source reductions across angular moments $a$. Suffers from strided memory accesses and lack of SIMD vectorization.
- **(b) Moment-Major Order**: Outer loop over angular moments $a$, updating contiguous group vectors with explicit SIMD vectorization, followed by a separate vector flux update.
- **(c) Group-Major Control**: Isolates loop order effects with identical temporary vector storage and separate flux update code.

### 2. Lock-Based Synchronization in Shared Regional Accumulation (Algorithm 2)
Concurrent OpenMP threads accumulate active ray segment flux changes $\Delta\psi_g$ into shared regional angular-moment tallies $C_{i,a,g}$:
- **Native OpenMP Locks (`omp_lock_t`)**: Standard OS-scheduled mutex with thread suspension.
- **Cache-Padded Spinlocks (`PaddedSpinLock`)**: Padded to 64 bytes (`alignas(64)`) to eliminate false sharing, spinning on atomic state.
- **MCS Queue Locks (`MCSLock`)**: Scalable queue lock (Mellor-Crummey and Scott) where each waiting thread spins strictly on a thread-private cache line, passing ownership via direct successor signaling.

---

## Quick Start

### Prerequisites
- Modern C++ compiler supporting C++17 (e.g., GCC >= 9.4, Clang >= 11, or MSVC)
- CMake >= 3.14
- OpenMP and POSIX Threads (`pthreads`)

### Build Instructions
```bash
git clone https://github.com/poet7/random-ray-loop-reorder-and-locks.git
cd random-ray-loop-reorder-and-locks
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run Tests and Benchmarks
To run the automated smoke test (quick verification in ~0.2 s):
```bash
cd build
ctest --output-on-failure
# Or directly via CLI:
./rr_benchmark --smoke
```

To run the full multi-threaded performance benchmark:
```bash
./rr_benchmark
```

### Command-Line Options
```text
Usage: ./rr_benchmark [options]
Options:
  --smoke              Run a fast smoke test (for quick verification and CI)
  --threads <N>        Set number of OpenMP worker threads (default: hardware max)
  --groups <G>         Set number of energy groups (default: 47)
  --moments <M>        Set number of spherical harmonics moments (default: 16 for P3)
  --segments <S>       Set number of segments for lock contention test (default: 2000000)
  --trials <T>         Set number of trials for local attenuation test (default: 5000000)
  -h, --help           Display usage message
```

---

## Representative Output
```text
======================================================================
  Multigroup Random-Ray Transport Optimization Benchmark
  Kernels: Loop Reordering (Alg. 1) & Lock Synchronization (Alg. 2)
======================================================================
Configuration:
  Energy Groups (G):       47
  Angular Moments (Nmom):  16 (P3 expansion)
  Attenuation Trials:      5000000
  Lock Benchmark Segments: 2000000
  OpenMP Threads:          32
----------------------------------------------------------------------

--- [Part 1] Loop Reordering in Local Flux Attenuation (Alg. 1) ---
  (a) Group-major time:            1769.460 ms  (baseline)
  (b) Moment-major time (SIMD):    1206.899 ms  [Speedup: 1.466x]
  (c) Group-major separate update: 2476.479 ms  [Speedup: 0.715x]
  Numerical difference epsilon_2 (Moment vs Group):   1.091e-16
  Numerical difference epsilon_2 (Control vs Group):  0.000e+00

--- [Part 2] Regional Lock Synchronization (Alg. 2, Threads: 32) ---
  Scenario: Coarse Mesh / High Contention (Nreg = 141)
    OpenMP Lock: 323.370 ms  (baseline)
    SpinLock:    322.009 ms  [Speedup vs OMP: 1.004x]
    MCS Lock:    340.839 ms  [Speedup vs OMP: 0.949x]
    Numerical diff (Spin vs OMP): 0.000
    Numerical diff (MCS  vs OMP): 0.000

  Scenario: Fine Mesh / Low Contention (Nreg = 1410)
    OpenMP Lock: 319.862 ms  (baseline)
    SpinLock:    328.815 ms  [Speedup vs OMP: 0.973x]
    MCS Lock:    342.380 ms  [Speedup vs OMP: 0.934x]
    Numerical diff (Spin vs OMP): 0.000
    Numerical diff (MCS  vs OMP): 0.000

======================================================================
  Verification Status: ALL TESTS PASSED
======================================================================
```

---

## Repository Layout
```text
random-ray-loop-reorder-and-locks/
├── CMakeLists.txt          # Portable CMake build configuration with CTest
├── README.md               # Documentation and reproducibility guide
├── LICENSE                 # MIT License
├── include/
│   ├── locks.hpp           # PaddedSpinLock, MCSLock, and OMPLock implementations
│   └── transport_kernels.hpp # Algorithm 1 and Algorithm 2 declarations
└── src/
    ├── transport_kernels.cpp # Core computational kernel implementations
    └── main.cpp            # Benchmark harness with verification and CLI options
```

---

## Citation
If you use these benchmark kernels or algorithms in your research, please cite our paper:
```bibtex
@article{li2026loopreordering,
  title={Loop reordering and lock-based synchronization in multigroup random-ray transport},
  author={Li, Shihong and Qin, Shuai and Tian, Xiangchun and Li, Jiacheng and Zhang, Qian},
  journal={Computer Physics Communications},
  year={2026}
}
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
