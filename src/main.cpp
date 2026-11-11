#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <omp.h>

#include "locks.hpp"
#include "transport_kernels.hpp"
#include "verification.hpp"

using namespace rr_benchmark;

// Synthetic benchmark configuration
struct BenchmarkConfig {
    int64_t n_groups = 47;           // 47 groups (BUGLE-96 structure)
    int64_t n_moments = 16;          // P3 scattering expansion (16 moments)
    int64_t n_segments = 2000000;    // Number of active segments for lock benchmark
    int64_t n_local_trials = 5000000;// Trials for local flux attenuation benchmark
    int64_t n_regions_coarse = 141;  // Coarse synthetic target mesh
    int64_t n_regions_fine = 1410;   // Fine synthetic target mesh
    int n_threads = 0;               // 0 = auto (all available hardware threads)
    bool smoke_test = false;         // Quick smoke test for verification / CI
};

int64_t positive_integer(const std::string& option, const std::string& text) {
    int64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value <= 0) {
        throw std::invalid_argument(option + " requires a positive integer");
    }
    return value;
}

bool verify(const std::string& label, const std::vector<double>& test,
            const std::vector<double>& reference) {
    const auto difference = compare_results(test, reference);
    std::cout << "    " << label << ": epsilon_2=" << std::scientific
              << difference.relative_l2 << ", epsilon_inf=" << difference.relative_linf
              << (difference.passes() ? " [PASS]\n" : " [FAIL]\n");
    return difference.passes();
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --smoke              Short checks: 1024 trials, 4096 requests, at most 4 workers\n"
              << "  --threads <N>        Set a positive worker count (default: OpenMP max, or <=4 for smoke)\n"
              << "  --groups <G>         Set energy groups, 1 <= G <= 256 (default: 47)\n"
              << "  --moments <M>        Set number of spherical harmonics moments (default: 16)\n"
              << "  --segments <S>       Set number of segments for lock contention test (default: 2000000)\n"
              << "  --trials <T>         Set number of trials for local attenuation test (default: 5000000)\n"
              << "  -h, --help           Display this help message\n";
}

int main(int argc, char** argv) try {
    BenchmarkConfig config;
    bool segments_specified = false, trials_specified = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--smoke") {
            config.smoke_test = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            if (arg != "--threads" && arg != "--groups" && arg != "--moments"
                    && arg != "--segments" && arg != "--trials") {
                throw std::invalid_argument("unknown option: " + arg);
            }
            if (i + 1 == argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            const int64_t value = positive_integer(arg, argv[++i]);
            if (arg == "--threads") {
                if (value > std::numeric_limits<int>::max()) {
                    throw std::invalid_argument("--threads exceeds the supported integer range");
                }
                config.n_threads = static_cast<int>(value);
            } else if (arg == "--groups") {
                config.n_groups = value;
            } else if (arg == "--moments") {
                config.n_moments = value;
            } else if (arg == "--segments") {
                config.n_segments = value;
                segments_specified = true;
            } else {
                config.n_local_trials = value;
                trials_specified = true;
            }
        }
    }

    if (config.n_groups > kMaxEnergyGroups) {
        throw std::invalid_argument("--groups must be between 1 and 256");
    }
    if (config.n_moments > std::numeric_limits<int64_t>::max()
            / config.n_regions_fine / config.n_groups) {
        throw std::invalid_argument("--moments is too large for the tally dimensions");
    }
    if (config.smoke_test) {
        if (!segments_specified) config.n_segments = 4096;
        if (!trials_specified) config.n_local_trials = 1024;
    }
    if (config.n_threads == 0) {
        config.n_threads = config.smoke_test
            ? std::min(4, omp_get_max_threads()) : omp_get_max_threads();
    }
    omp_set_num_threads(config.n_threads);

    std::cout << "======================================================================\n";
    std::cout << "  Multigroup Random-Ray Transport Optimization Benchmark\n";
    std::cout << "  Kernels: Loop Reordering (Alg. 1) & Lock Synchronization (Alg. 2)\n";
    std::cout << "======================================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Energy Groups (G):       " << config.n_groups << "\n";
    std::cout << "  Angular Moments (Nmom):  " << config.n_moments << "\n";
    std::cout << "  Attenuation Trials:      " << config.n_local_trials << "\n";
    std::cout << "  Lock Benchmark Segments: " << config.n_segments << "\n";
    std::cout << "  OpenMP Threads:          " << config.n_threads << "\n";
    std::cout << "  Numerical tolerance:     " << kVerificationTolerance << " (relative L2 and Linf)\n";
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
    std::vector<double> psi_out_moment_control(G, 0.0), delta_moment_control(G, 0.0);

    // 1(a) Group-major order (Algorithm 1(a))
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < config.n_local_trials; ++i) {
        attenuate_group_major(G, Nmom, q_tilde.data(), sigma_t.data(), R.data(),
                               seg_length, psi_in.data(), psi_out_group.data(), delta_group.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    double time_group_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 1(b) Moment-major order (Algorithm 1(b) with SIMD vectorization)
    t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < config.n_local_trials; ++i) {
        attenuate_moment_major(G, Nmom, q_tilde.data(), sigma_t.data(), R.data(),
                                seg_length, psi_in.data(), psi_out_moment.data(), delta_moment.data());
    }
    t1 = std::chrono::steady_clock::now();
    double time_moment_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 1(c) Control loop (Group-major with separate vector flux update)
    t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < config.n_local_trials; ++i) {
        attenuate_group_major_separate_update(G, Nmom, q_tilde.data(), sigma_t.data(), R.data(),
                                              seg_length, psi_in.data(), psi_out_control.data(), delta_control.data());
    }
    t1 = std::chrono::steady_clock::now();
    double time_control_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Matched loop-order comparison: both source sums omit explicit SIMD.
    t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < config.n_local_trials; ++i) {
        attenuate_moment_major_control(G, Nmom, q_tilde.data(), sigma_t.data(), R.data(),
                                      seg_length, psi_in.data(), psi_out_moment_control.data(),
                                      delta_moment_control.data());
    }
    t1 = std::chrono::steady_clock::now();
    double time_moment_control_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double speedup_moment = (time_moment_ms > 0.0) ? (time_group_ms / time_moment_ms) : 1.0;
    double speedup_control = (time_control_ms > 0.0) ? (time_group_ms / time_control_ms) : 1.0;
    double loop_order_speedup = (time_moment_control_ms > 0.0)
        ? (time_control_ms / time_moment_control_ms) : 1.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  (a) Group-major time:            " << time_group_ms << " ms  (baseline)\n";
    std::cout << "  (b) Moment-major time (SIMD):    " << time_moment_ms << " ms  [Speedup: " << speedup_moment << "x]\n";
    std::cout << "  (c) Group-major separate update: " << time_control_ms << " ms  [Speedup: " << speedup_control << "x]\n";
    std::cout << "  (d) Moment-major control:       " << time_moment_control_ms << " ms\n";
    std::cout << "  Controlled loop-order speedup (c)/(d): " << loop_order_speedup << "x\n";

    bool all_checks_passed = true;
    all_checks_passed &= verify("(b)/(a) angular-flux changes", delta_moment, delta_group);
    all_checks_passed &= verify("(b)/(a) outgoing fluxes", psi_out_moment, psi_out_group);
    all_checks_passed &= verify("(c)/(a) angular-flux changes", delta_control, delta_group);
    all_checks_passed &= verify("(c)/(a) outgoing fluxes", psi_out_control, psi_out_group);
    all_checks_passed &= verify("(d)/(c) angular-flux changes", delta_moment_control, delta_control);
    all_checks_passed &= verify("(d)/(c) outgoing fluxes", psi_out_moment_control, psi_out_control);

    // ========================================================================
    // PART 2: Algorithm 2 - Regional Lock Synchronization
    // ========================================================================
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "--- [Part 2] Regional Lock Synchronization (Alg. 2, Threads: " << config.n_threads << ") ---\n";

    auto run_lock_benchmark = [&](int64_t n_regions, const std::string& desc) {
        std::cout << "  Scenario: " << desc << " (Nreg = " << n_regions << ")\n";

        // Generate synthetic regional target sequence
        std::vector<int64_t> region_targets(config.n_segments);
        std::vector<int64_t> region_counts(n_regions, 0);
        std::uniform_int_distribution<int64_t> reg_dist(0, n_regions - 1);
        for (auto& r : region_targets) {
            r = reg_dist(rng);
            ++region_counts[r];
        }

        // 2(a) Native OpenMP Locks
        std::vector<double> tally_omp(n_regions * Nmom * G, 0.0);
        std::vector<OMPLock> omp_locks(n_regions);

        t0 = std::chrono::steady_clock::now();
        #pragma omp parallel num_threads(config.n_threads)
        {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < config.n_segments; ++i) {
                int64_t reg = region_targets[i];
                accumulate_moments_openmp(G, Nmom, R.data(), delta_moment.data(),
                                         &tally_omp[reg * Nmom * G], omp_locks[reg]);
            }
        }
        t1 = std::chrono::steady_clock::now();
        double time_omp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2(b) Padded SpinLocks
        std::vector<double> tally_spin(n_regions * Nmom * G, 0.0);
        std::vector<PaddedSpinLock> spin_locks(n_regions);

        t0 = std::chrono::steady_clock::now();
        #pragma omp parallel num_threads(config.n_threads)
        {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < config.n_segments; ++i) {
                int64_t reg = region_targets[i];
                accumulate_moments_spinlock(G, Nmom, R.data(), delta_moment.data(),
                                           &tally_spin[reg * Nmom * G], spin_locks[reg]);
            }
        }
        t1 = std::chrono::steady_clock::now();
        double time_spin_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2(c) MCS Queue Locks
        std::vector<double> tally_mcs(n_regions * Nmom * G, 0.0);
        std::vector<MCSLock> mcs_locks(n_regions);

        t0 = std::chrono::steady_clock::now();
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
        t1 = std::chrono::steady_clock::now();
        double time_mcs_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double speedup_spin = (time_spin_ms > 0.0) ? (time_omp_ms / time_spin_ms) : 1.0;
        double speedup_mcs  = (time_mcs_ms > 0.0)  ? (time_omp_ms / time_mcs_ms)  : 1.0;

        std::cout << "    OpenMP Lock: " << time_omp_ms << " ms  (baseline)\n";
        std::cout << "    SpinLock:    " << time_spin_ms << " ms  [Speedup vs OMP: "
                  << speedup_spin << "x]\n";
        std::cout << "    MCS Lock:    " << time_mcs_ms << " ms  [Speedup vs OMP: "
                  << speedup_mcs << "x]\n";
        // Each synthetic request adds the same moment vector; counts give an
        // independent reference that can detect lost updates in every lock.
        std::vector<double> expected(n_regions * Nmom * G, 0.0);
        for (int64_t region = 0; region < n_regions; ++region) {
            for (int64_t a = 0; a < Nmom; ++a) {
                for (int64_t g = 0; g < G; ++g) {
                    expected[(region * Nmom + a) * G + g] =
                        region_counts[region] * (R[a] * delta_moment[g]);
                }
            }
        }
        all_checks_passed &= verify("OpenMP/expected tally", tally_omp, expected);
        all_checks_passed &= verify("Spinlock/expected tally", tally_spin, expected);
        all_checks_passed &= verify("MCS/expected tally", tally_mcs, expected);
        all_checks_passed &= verify("Spinlock/OpenMP tally", tally_spin, tally_omp);
        all_checks_passed &= verify("MCS/OpenMP tally", tally_mcs, tally_omp);
        std::cout << std::fixed << std::setprecision(3);
    };

    run_lock_benchmark(config.n_regions_coarse, "Coarse target mesh");
    run_lock_benchmark(config.n_regions_fine,   "Fine target mesh");

    std::cout << "======================================================================\n";
    std::cout << "  Verification Status: "
              << (all_checks_passed ? "ALL CHECKS PASSED\n" : "CHECKS FAILED\n");
    std::cout << "======================================================================\n";

    return all_checks_passed ? EXIT_SUCCESS : EXIT_FAILURE;
} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return EXIT_FAILURE;
}
