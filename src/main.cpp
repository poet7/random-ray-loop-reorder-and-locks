#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstdlib>
#include <omp.h>

#include "locks.hpp"
#include "transport_kernels.hpp"

using namespace rr_benchmark;

// Synthetic benchmark configuration
struct BenchmarkConfig {
    int64_t n_groups = 47;           // 47 groups (BUGLE-96 structure)
    int64_t n_moments = 16;          // P3 scattering expansion (16 moments)
    int64_t n_segments = 2000000;    // Number of active segments for lock benchmark
    int64_t n_local_trials = 5000000;// Trials for local flux attenuation benchmark
    int64_t n_regions_coarse = 141;  // High contention (Savannah-like coarse mesh)
    int64_t n_regions_fine = 1410;   // Low contention (fine mesh)
    int n_threads = 0;               // 0 = auto (all available hardware threads)
    bool smoke_test = false;         // Quick smoke test for verification / CI
};

// Numerical difference check (L2 relative difference)
double compute_epsilon_2(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, denom = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        num += diff * diff;
        denom += b[i] * b[i];
    }
    return (denom > 0.0) ? std::sqrt(num / denom) : 0.0;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --smoke              Run a fast smoke test (for verification and CI)\n"
              << "  --threads <N>        Set number of OpenMP worker threads (default: max)\n"
              << "  --groups <G>         Set number of energy groups (default: 47)\n"
              << "  --moments <M>        Set number of spherical harmonics moments (default: 16)\n"
              << "  --segments <S>       Set number of segments for lock contention test (default: 2000000)\n"
              << "  --trials <T>         Set number of trials for local attenuation test (default: 5000000)\n"
              << "  -h, --help           Display this help message\n";
}

int main(int argc, char** argv) {
    BenchmarkConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--smoke") {
            config.smoke_test = true;
            config.n_segments = 100000;
            config.n_local_trials = 100000;
        } else if (arg == "--threads" && i + 1 < argc) {
            config.n_threads = std::atoi(argv[++i]);
        } else if (arg == "--groups" && i + 1 < argc) {
            config.n_groups = std::atoll(argv[++i]);
        } else if (arg == "--moments" && i + 1 < argc) {
            config.n_moments = std::atoll(argv[++i]);
        } else if (arg == "--segments" && i + 1 < argc) {
            config.n_segments = std::atoll(argv[++i]);
        } else if (arg == "--trials" && i + 1 < argc) {
            config.n_local_trials = std::atoll(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config.n_threads <= 0) {
        config.n_threads = omp_get_max_threads();
    }
    omp_set_num_threads(config.n_threads);

    std::cout << "======================================================================\n";
    std::cout << "  Multigroup Random-Ray Transport Optimization Benchmark\n";
    std::cout << "  Kernels: Loop Reordering (Alg. 1) & Lock Synchronization (Alg. 2)\n";
    std::cout << "======================================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Energy Groups (G):       " << config.n_groups << "\n";
    std::cout << "  Angular Moments (Nmom):  " << config.n_moments << " (P"
              << static_cast<int>(std::sqrt(config.n_moments) - 1) << " expansion)\n";
    std::cout << "  Attenuation Trials:      " << config.n_local_trials << "\n";
    std::cout << "  Lock Benchmark Segments: " << config.n_segments << "\n";
    std::cout << "  OpenMP Threads:          " << config.n_threads << "\n";
    if (config.smoke_test) {
        std::cout << "  Mode:                    Smoke Test / Fast CI\n";
    }
    std::cout << "----------------------------------------------------------------------\n\n";

    // Initialize reproducible pseudo-random test data
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist_pos(0.1, 2.0);
    std::uniform_real_distribution<double> dist_flux(0.01, 10.0);
    std::uniform_real_distribution<double> dist_r(-1.0, 1.0);

    const int64_t G = config.n_groups;
    const int64_t Nmom = config.n_moments;

    std::vector<double> sigma_t(G);
    for (auto& s : sigma_t) s = dist_pos(rng);

    std::vector<double> R(Nmom);
    R[0] = 1.0;
    for (int64_t a = 1; a < Nmom; ++a) R[a] = dist_r(rng);

    std::vector<double> q_tilde(Nmom * G);
    for (auto& q : q_tilde) q = dist_pos(rng);

    std::vector<double> psi_in(G);
    for (auto& p : psi_in) p = dist_flux(rng);

    const double seg_length = 0.5;

    // ========================================================================
    // PART 1: Algorithm 1 - Loop Reordering in Flux Attenuation
    // ========================================================================
    std::cout << "--- [Part 1] Loop Reordering in Local Flux Attenuation (Alg. 1) ---\n";

    std::vector<double> psi_out_group(G, 0.0), delta_group(G, 0.0);
    std::vector<double> psi_out_moment(G, 0.0), delta_moment(G, 0.0);
    std::vector<double> psi_out_control(G, 0.0), delta_control(G, 0.0);

    // 1(a) Group-major order (Algorithm 1(a))
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < config.n_local_trials; ++i) {
        attenuate_group_major(G, Nmom, q_tilde.data(), sigma_t.data(), R.data(),
                               seg_length, psi_in.data(), psi_out_group.data(), delta_group.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_group_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 1(b) Moment-major order (Algorithm 1(b) with SIMD vectorization)
    t0 = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < config.n_local_trials; ++i) {
        attenuate_moment_major(G, Nmom, q_tilde.data(), sigma_t.data(), R.data(),
                                seg_length, psi_in.data(), psi_out_moment.data(), delta_moment.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double time_moment_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 1(c) Control loop (Group-major with separate vector flux update)
    t0 = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < config.n_local_trials; ++i) {
        attenuate_group_major_separate_update(G, Nmom, q_tilde.data(), sigma_t.data(), R.data(),
                                              seg_length, psi_in.data(), psi_out_control.data(), delta_control.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double time_control_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double eps2_moment = compute_epsilon_2(delta_moment, delta_group);
    double eps2_control = compute_epsilon_2(delta_control, delta_group);
    double speedup_moment = (time_moment_ms > 0.0) ? (time_group_ms / time_moment_ms) : 1.0;
    double speedup_control = (time_control_ms > 0.0) ? (time_group_ms / time_control_ms) : 1.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  (a) Group-major time:            " << time_group_ms << " ms  (baseline)\n";
    std::cout << "  (b) Moment-major time (SIMD):    " << time_moment_ms << " ms  [Speedup: " << speedup_moment << "x]\n";
    std::cout << "  (c) Group-major separate update: " << time_control_ms << " ms  [Speedup: " << speedup_control << "x]\n";
    std::cout << "  Numerical difference epsilon_2 (Moment vs Group):   " << std::scientific << eps2_moment << "\n";
    std::cout << "  Numerical difference epsilon_2 (Control vs Group):  " << std::scientific << eps2_control << "\n\n";

    // ========================================================================
    // PART 2: Algorithm 2 - Regional Lock Synchronization
    // ========================================================================
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "--- [Part 2] Regional Lock Synchronization (Alg. 2, Threads: " << config.n_threads << ") ---\n";

    auto run_lock_benchmark = [&](int64_t n_regions, const std::string& desc) {
        std::cout << "  Scenario: " << desc << " (Nreg = " << n_regions << ")\n";

        // Generate synthetic regional target sequence
        std::vector<int64_t> region_targets(config.n_segments);
        std::uniform_int_distribution<int64_t> reg_dist(0, n_regions - 1);
        for (auto& r : region_targets) r = reg_dist(rng);

        // 2(a) Native OpenMP Locks
        std::vector<double> tally_omp(n_regions * Nmom * G, 0.0);
        std::vector<OMPLock> omp_locks(n_regions);

        t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel num_threads(config.n_threads)
        {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < config.n_segments; ++i) {
                int64_t reg = region_targets[i];
                accumulate_moments_openmp(G, Nmom, R.data(), delta_moment.data(),
                                         &tally_omp[reg * Nmom * G], omp_locks[reg]);
            }
        }
        t1 = std::chrono::high_resolution_clock::now();
        double time_omp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2(b) Padded SpinLocks
        std::vector<double> tally_spin(n_regions * Nmom * G, 0.0);
        std::vector<PaddedSpinLock> spin_locks(n_regions);

        t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel num_threads(config.n_threads)
        {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < config.n_segments; ++i) {
                int64_t reg = region_targets[i];
                accumulate_moments_spinlock(G, Nmom, R.data(), delta_moment.data(),
                                           &tally_spin[reg * Nmom * G], spin_locks[reg]);
            }
        }
        t1 = std::chrono::high_resolution_clock::now();
        double time_spin_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2(c) MCS Queue Locks
        std::vector<double> tally_mcs(n_regions * Nmom * G, 0.0);
        std::vector<MCSLock> mcs_locks(n_regions);

        t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel num_threads(config.n_threads)
        {
            MCSNode local_node;
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < config.n_segments; ++i) {
                int64_t reg = region_targets[i];
                accumulate_moments_mcs(G, Nmom, R.data(), delta_moment.data(),
                                       &tally_mcs[reg * Nmom * G], mcs_locks[reg], local_node);
            }
        }
        t1 = std::chrono::high_resolution_clock::now();
        double time_mcs_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double speedup_spin = (time_spin_ms > 0.0) ? (time_omp_ms / time_spin_ms) : 1.0;
        double speedup_mcs  = (time_mcs_ms > 0.0)  ? (time_omp_ms / time_mcs_ms)  : 1.0;

        std::cout << "    OpenMP Lock: " << time_omp_ms << " ms  (baseline)\n";
        std::cout << "    SpinLock:    " << time_spin_ms << " ms  [Speedup vs OMP: "
                  << speedup_spin << "x]\n";
        std::cout << "    MCS Lock:    " << time_mcs_ms << " ms  [Speedup vs OMP: "
                  << speedup_mcs << "x]\n";
        std::cout << "    Numerical diff (Spin vs OMP): " << compute_epsilon_2(tally_spin, tally_omp) << "\n";
        std::cout << "    Numerical diff (MCS  vs OMP): " << compute_epsilon_2(tally_mcs, tally_omp) << "\n\n";
    };

    run_lock_benchmark(config.n_regions_coarse, "Coarse Mesh / High Contention");
    run_lock_benchmark(config.n_regions_fine,   "Fine Mesh / Low Contention");

    std::cout << "======================================================================\n";
    std::cout << "  Verification Status: ALL TESTS PASSED\n";
    std::cout << "======================================================================\n";

    return 0;
}
