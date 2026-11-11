#ifndef RANDOM_RAY_VERIFICATION_HPP
#define RANDOM_RAY_VERIFICATION_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace rr_benchmark {

constexpr double kVerificationTolerance = 1.0e-8;

struct NumericalDifference {
    double relative_l2;
    double relative_linf;

    bool passes(double tolerance = kVerificationTolerance) const {
        return tolerance >= 0.0 && std::isfinite(relative_l2)
            && std::isfinite(relative_linf)
            && relative_l2 <= tolerance && relative_linf <= tolerance;
    }
};

inline NumericalDifference compare_results(
    const std::vector<double>& test,
    const std::vector<double>& reference
) {
    const double infinity = std::numeric_limits<double>::infinity();
    if (test.size() != reference.size()) {
        return {infinity, infinity};
    }

    long double difference_norm = 0.0L, reference_norm = 0.0L;
    long double difference_max = 0.0L, reference_max = 0.0L;
    for (std::size_t i = 0; i < test.size(); ++i) {
        if (!std::isfinite(test[i]) || !std::isfinite(reference[i])) {
            return {infinity, infinity};
        }
        const long double difference = static_cast<long double>(test[i]) - reference[i];
        difference_norm = std::hypot(difference_norm, difference);
        reference_norm = std::hypot(reference_norm, static_cast<long double>(reference[i]));
        difference_max = std::max(difference_max, std::abs(difference));
        reference_max = std::max(reference_max, std::abs(static_cast<long double>(reference[i])));
    }

    // An all-zero reference passes only if the test is also exactly zero.
    const auto relative = [infinity](long double difference, long double reference_value) {
        return reference_value > 0.0L ? static_cast<double>(difference / reference_value)
            : (difference == 0.0L ? 0.0 : infinity);
    };
    return {relative(difference_norm, reference_norm), relative(difference_max, reference_max)};
}

} // namespace rr_benchmark

#endif // RANDOM_RAY_VERIFICATION_HPP
