#include "transport_kernels.hpp"
#include "verification.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rr_benchmark;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_verification() {
    require(compare_results({1.0, 2.0}, {1.0, 2.0}).passes(), "identical results rejected");
    require(compare_results({0.0, 0.0}, {0.0, 0.0}).passes(), "identical zeros rejected");
    require(!compare_results({1.0}, {0.0}).passes(), "nonzero result with zero reference accepted");
    require(!compare_results({1.0}, {1.0, 2.0}).passes(), "different vector sizes accepted");
    require(compare_results({1.0 + 1.0e-10}, {1.0}).passes(), "within-tolerance result rejected");
    require(!compare_results({1.0 + 1.0e-6}, {1.0}).passes(), "out-of-tolerance result accepted");
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
        require(!compare_results({invalid}, {1.0}).passes(), "nonfinite test value accepted");
        require(!compare_results({1.0}, {invalid}).passes(), "nonfinite reference accepted");
    }

    std::vector<double> reference(10000, 1.0), test = reference;
    test[0] += 1.0e-7;
    const auto difference = compare_results(test, reference);
    require(difference.relative_l2 < kVerificationTolerance
            && difference.relative_linf > kVerificationTolerance && !difference.passes(),
            "Linf tolerance is not enforced independently of L2");
}

void test_attenuation() {
    const auto kernels = {attenuate_group_major, attenuate_moment_major,
                          attenuate_group_major_separate_update, attenuate_moment_major_control};
    for (int64_t groups : {int64_t{1}, int64_t{47}, kMaxEnergyGroups}) {
        for (int64_t moments : {1, 4, 16}) {
            std::vector<double> coefficients(moments * groups), basis(moments);
            std::vector<double> sigma(groups), incoming(groups);
            for (int64_t a = 0; a < moments; ++a) {
                basis[a] = (a % 2 == 0 ? 1.0 : -1.0) / (a + 1);
                for (int64_t g = 0; g < groups; ++g) {
                    coefficients[a * groups + g] = 0.2 + 0.01 * a + 0.001 * g;
                }
            }
            for (int64_t g = 0; g < groups; ++g) {
                sigma[g] = 0.1 + 0.01 * g;
                incoming[g] = 0.5 + 0.02 * g;
            }

            for (double length : {0.0, 1.0e-8, 0.5}) {
                std::vector<double> expected_out(groups), expected_delta(groups);
                for (int64_t g = 0; g < groups; ++g) {
                    long double source = 0.0L;
                    for (int64_t a = 0; a < moments; ++a) {
                        source += static_cast<long double>(basis[a]) * coefficients[a * groups + g];
                    }
                    const long double attenuation = -std::expm1(
                        -static_cast<long double>(sigma[g]) * length);
                    expected_delta[g] = (incoming[g] - source) * attenuation;
                    expected_out[g] = incoming[g] - expected_delta[g];
                }
                for (auto kernel : kernels) {
                    std::vector<double> outgoing(groups), delta(groups);
                    kernel(groups, moments, coefficients.data(), sigma.data(), basis.data(),
                           length, incoming.data(), outgoing.data(), delta.data());
                    require(compare_results(outgoing, expected_out).passes(), "incorrect outgoing flux");
                    require(compare_results(delta, expected_delta).passes(), "incorrect angular-flux change");
                }
            }
        }
    }

    for (auto kernel : kernels) {
        for (int64_t groups : {-1, 0, 257}) {
            bool rejected = false;
            try {
                kernel(groups, 1, nullptr, nullptr, nullptr, 1.0, nullptr, nullptr, nullptr);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "invalid group count reached the attenuation kernel");
        }
        bool rejected = false;
        try {
            kernel(1, 0, nullptr, nullptr, nullptr, 1.0, nullptr, nullptr, nullptr);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "zero moment count reached the attenuation kernel");
    }
}

} // namespace

int main() try {
    test_verification();
    test_attenuation();
    std::cout << "Kernel and verification tests passed.\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Test failed: " << error.what() << "\n";
    return 1;
}
