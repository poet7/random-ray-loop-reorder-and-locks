#include "transport_kernels.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>

namespace rr_benchmark {

namespace {

void validate_dimensions(int64_t G, int64_t Nmom) {
    if (G < 1 || G > kMaxEnergyGroups || Nmom < 1) {
        throw std::invalid_argument("attenuation requires 1 <= groups <= 256 and moments >= 1");
    }
}

void advance_group_fluxes(
    int64_t G,
    const double* __restrict__ q_tilde_dir,
    const double* __restrict__ sigma_t,
    double length,
    const double* __restrict__ psi_in,
    double* __restrict__ psi_out,
    double* __restrict__ delta_psi
) {
    #pragma omp simd
    for (int64_t g = 0; g < G; ++g) {
        double tau = sigma_t[g] * length;
        double A = -std::expm1(-tau);
        double d_psi = (psi_in[g] - q_tilde_dir[g]) * A;
        delta_psi[g] = d_psi;
        psi_out[g] = psi_in[g] - d_psi;
    }
}

} // namespace

// ============================================================================
// Algorithm 1: Loop Reordering in Local Flux Attenuation
// ============================================================================

// (a) Group-major order
void attenuate_group_major(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ q_tilde,
    const double* __restrict__ sigma_t,
    const double* __restrict__ R,
    double length,
    const double* __restrict__ psi_in,
    double* __restrict__ psi_out,
    double* __restrict__ delta_psi
) {
    validate_dimensions(G, Nmom);
    for (int64_t g = 0; g < G; ++g) {
        // Scalar reduction across moments for group g (stride G access)
        double q_tilde_dir = 0.0;
        for (int64_t a = 0; a < Nmom; ++a) {
            q_tilde_dir += R[a] * q_tilde[a * G + g];
        }

        // Immediate flux update
        double tau = sigma_t[g] * length;
        double A = -std::expm1(-tau);
        double d_psi = (psi_in[g] - q_tilde_dir) * A;
        delta_psi[g] = d_psi;
        psi_out[g] = psi_in[g] - d_psi;
    }
}

// (b) Moment-major order
void attenuate_moment_major(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ q_tilde,
    const double* __restrict__ sigma_t,
    const double* __restrict__ R,
    double length,
    const double* __restrict__ psi_in,
    double* __restrict__ psi_out,
    double* __restrict__ delta_psi
) {
    validate_dimensions(G, Nmom);
    alignas(64) double q_tilde_dir[kMaxEnergyGroups] = {0.0};

    // Moment outer loop: contiguous group access permitting SIMD vectorization
    for (int64_t a = 0; a < Nmom; ++a) {
        const double r_a = R[a];
        const double* __restrict__ q_row = &q_tilde[a * G];

        #pragma omp simd
        for (int64_t g = 0; g < G; ++g) {
            q_tilde_dir[g] += r_a * q_row[g];
        }
    }

    advance_group_fluxes(G, q_tilde_dir, sigma_t, length, psi_in, psi_out, delta_psi);
}

// (c) Group-major order with separate vector flux update (loop-order control)
void attenuate_group_major_separate_update(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ q_tilde,
    const double* __restrict__ sigma_t,
    const double* __restrict__ R,
    double length,
    const double* __restrict__ psi_in,
    double* __restrict__ psi_out,
    double* __restrict__ delta_psi
) {
    validate_dimensions(G, Nmom);
    alignas(64) double q_tilde_dir[kMaxEnergyGroups] = {0.0};

    for (int64_t g = 0; g < G; ++g) {
        double sum = 0.0;
        for (int64_t a = 0; a < Nmom; ++a) {
            sum += R[a] * q_tilde[a * G + g];
        }
        q_tilde_dir[g] = sum;
    }

    advance_group_fluxes(G, q_tilde_dir, sigma_t, length, psi_in, psi_out, delta_psi);
}

void attenuate_moment_major_control(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ q_tilde,
    const double* __restrict__ sigma_t,
    const double* __restrict__ R,
    double length,
    const double* __restrict__ psi_in,
    double* __restrict__ psi_out,
    double* __restrict__ delta_psi
) {
    validate_dimensions(G, Nmom);
    alignas(64) double q_tilde_dir[kMaxEnergyGroups] = {0.0};

    for (int64_t a = 0; a < Nmom; ++a) {
        const double r_a = R[a];
        const double* __restrict__ q_row = &q_tilde[a * G];
        for (int64_t g = 0; g < G; ++g) {
            q_tilde_dir[g] += r_a * q_row[g];
        }
    }

    advance_group_fluxes(G, q_tilde_dir, sigma_t, length, psi_in, psi_out, delta_psi);
}

// ============================================================================
// Algorithm 2: Regional Moment Accumulation with Lock Synchronization
// ============================================================================

#ifdef _OPENMP
void accumulate_moments_openmp(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ R,
    const double* __restrict__ delta_psi,
    double* __restrict__ regional_tally,
    OMPLock& lock
) {
    lock.acquire();
    for (int64_t a = 0; a < Nmom; ++a) {
        const double r_a = R[a];
        double* __restrict__ tally_row = &regional_tally[a * G];
        #pragma omp simd
        for (int64_t g = 0; g < G; ++g) {
            tally_row[g] += r_a * delta_psi[g];
        }
    }
    lock.release();
}
#endif

void accumulate_moments_spinlock(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ R,
    const double* __restrict__ delta_psi,
    double* __restrict__ regional_tally,
    PaddedSpinLock& lock
) {
    lock.acquire();
    for (int64_t a = 0; a < Nmom; ++a) {
        const double r_a = R[a];
        double* __restrict__ tally_row = &regional_tally[a * G];
        #pragma omp simd
        for (int64_t g = 0; g < G; ++g) {
            tally_row[g] += r_a * delta_psi[g];
        }
    }
    lock.release();
}

void accumulate_moments_mcs(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ R,
    const double* __restrict__ delta_psi,
    double* __restrict__ regional_tally,
    MCSLock& lock,
    MCSNode& node
) {
    lock.acquire(&node);
    for (int64_t a = 0; a < Nmom; ++a) {
        const double r_a = R[a];
        double* __restrict__ tally_row = &regional_tally[a * G];
        #pragma omp simd
        for (int64_t g = 0; g < G; ++g) {
            tally_row[g] += r_a * delta_psi[g];
        }
    }
    lock.release(&node);
}

} // namespace rr_benchmark
