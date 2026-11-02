#ifndef RANDOM_RAY_LOCKS_HPP
#define RANDOM_RAY_LOCKS_HPP

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace rr_benchmark {

// =========================================================
// Portable CPU Relaxation
// =========================================================
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
static inline void cpu_relax() {
    _mm_pause();
}
#elif defined(__aarch64__) || defined(_M_ARM64)
static inline void cpu_relax() {
    asm volatile("yield" ::: "memory");
}
#else
static inline void cpu_relax() {
    asm volatile("" ::: "memory");
}
#endif

// =========================================================
// Cache line alignment to avoid false sharing
// =========================================================
constexpr std::size_t CACHE_LINE_SIZE = 64;

// =========================================================
// 1. Padded Spinlock (pthread_spinlock_t aligned to 64 bytes)
// =========================================================
struct alignas(CACHE_LINE_SIZE) PaddedSpinLock {
    pthread_spinlock_t lock;

    PaddedSpinLock() {
        pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE);
    }
    ~PaddedSpinLock() {
        pthread_spin_destroy(&lock);
    }

    void acquire() {
        pthread_spin_lock(&lock);
    }
    void release() {
        pthread_spin_unlock(&lock);
    }
};

// =========================================================
// 2. MCS Queue Lock (Mellor-Crummey and Scott)
// =========================================================
struct MCSNode {
    std::atomic<MCSNode*> next{nullptr};
    std::atomic<bool> waiting{false};
};

struct alignas(CACHE_LINE_SIZE) MCSLock {
    std::atomic<MCSNode*> tail{nullptr};

    void acquire(MCSNode* my_node) {
        my_node->next.store(nullptr, std::memory_order_relaxed);
        my_node->waiting.store(true, std::memory_order_relaxed);

        MCSNode* prev = tail.exchange(my_node, std::memory_order_acq_rel);
        if (prev != nullptr) {
            prev->next.store(my_node, std::memory_order_release);
            while (my_node->waiting.load(std::memory_order_acquire)) {
                cpu_relax();
            }
        } else {
            my_node->waiting.store(false, std::memory_order_relaxed);
        }
    }

    void release(MCSNode* my_node) {
        MCSNode* next = my_node->next.load(std::memory_order_acquire);
        if (next == nullptr) {
            MCSNode* expected = my_node;
            if (tail.compare_exchange_strong(expected, nullptr,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
                return;
            }
            do {
                cpu_relax();
                next = my_node->next.load(std::memory_order_acquire);
            } while (next == nullptr);
        }
        next->waiting.store(false, std::memory_order_release);
    }
};

// =========================================================
// 3. Native OpenMP Lock Wrapper
// =========================================================
#ifdef _OPENMP
struct OMPLock {
    omp_lock_t lock;
    OMPLock() { omp_init_lock(&lock); }
    ~OMPLock() { omp_destroy_lock(&lock); }
    void acquire() { omp_set_lock(&lock); }
    void release() { omp_unset_lock(&lock); }
};
#endif

} // namespace rr_benchmark

#endif // RANDOM_RAY_LOCKS_HPP
