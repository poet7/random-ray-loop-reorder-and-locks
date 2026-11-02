#ifndef RANDOM_RAY_TRANSPORT_KERNELS_HPP
#define RANDOM_RAY_TRANSPORT_KERNELS_HPP

#pragma once

#include <cstdint>
#include <vector>
#include "locks.hpp"

namespace rr_benchmark {

// ============================================================================
// Algorithm 1: Loop Reordering in Local Flux Attenuation
// ============================================================================

// (a) Group-major order: Group outer loop, scalar moment sum, immediate flux update
void attenuate_group_major(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ q_tilde, // [Nmom, G] row-major: q_tilde[a * G + g]
    const double* __restrict__ sigma_t, // [G]
    const double* __restrict__ R,       // [Nmom]
    double length,
    const double* __restrict__ psi_in,  // [G]
    double* __restrict__ psi_out,       // [G]
    double* __restrict__ delta_psi      // [G]
);

// (b) Moment-major order: Moment outer loop, vector source sum with SIMD, separate flux update
void attenuate_moment_major(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ q_tilde, // [Nmom, G] row-major: q_tilde[a * G + g]
    const double* __restrict__ sigma_t, // [G]
    const double* __restrict__ R,       // [Nmom]
    double length,
    const double* __restrict__ psi_in,  // [G]
    double* __restrict__ psi_out,       // [G]
    double* __restrict__ delta_psi      // [G]
);

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
);

// ============================================================================
// Algorithm 2: Regional Moment Accumulation with Lock Synchronization
// ============================================================================

enum class LockStrategy {
    OpenMP,
    Spinlock,
    MCS
};

// Accumulate delta_psi into regional moment tally C_i
// C_i layout: [Nmom, G] contiguous: C[a * G + g]
void accumulate_moments_openmp(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ R,
    const double* __restrict__ delta_psi,
    double* __restrict__ regional_tally,
    OMPLock& lock
);

void accumulate_moments_spinlock(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ R,
    const double* __restrict__ delta_psi,
    double* __restrict__ regional_tally,
    PaddedSpinLock& lock
);

void accumulate_moments_mcs(
    int64_t G,
    int64_t Nmom,
    const double* __restrict__ R,
    const double* __restrict__ delta_psi,
    double* __restrict__ regional_tally,
    MCSLock& lock,
    MCSNode& node
);

} // namespace rr_benchmark

#endif // RANDOM_RAY_TRANSPORT_KERNELS_HPP
