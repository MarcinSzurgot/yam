#pragma once

#include "MLPStructure.hpp"
#include "Utils.hpp"

#include <functional>

namespace yam {

struct MLPerceptron {

    using activation_function = std::function<float*(std::span<const float>, float*)>;

    template<Integers Topology>
    MLPerceptron(
        Topology&& topology,
        bool bias,
        activation_function activation
    ) : MLPerceptron(true, topology, bias, activation) {}

    template<std::integral I>
    MLPerceptron(
        std::initializer_list<I> topology,
        bool bias,
        activation_function activation
    ) : MLPerceptron(true, topology, bias, activation) {}

    auto forward(std::span<const float> input) -> std::span<const float> {
        auto neurons = structure_.neurons().values();
        auto lower = neurons.begin().base();
        auto upper = std::ranges::copy(input, lower).out;
        auto last  = neurons.end().base();
        auto weight = structure_.weights().values().begin();
        auto bias = structure_.biases().values().begin();

        std::fill(upper, last, 0);

        for (const auto [lc, uc] : adjacent(structure_.topology())) {
            for (auto l = 0; l < lc; ++l) {
                for (auto u = 0; u < uc; ++u) {
                    upper[u] += *weight++ * lower[l];
                }
            }

            if (bias != structure_.biases().values().end()) {
                for (auto u = 0; u < uc; ++u) {
                    upper[u] += *bias++;
                }
            }

            lower = upper;
            upper = activation_({upper, upper + uc}, upper);
        }

        return {lower, last};
    }

    auto structure() const -> const MLPStructure& { return structure_; }
    auto structure()       ->       MLPStructure& { return structure_; }

private:
    template<Integers Topology>
    MLPerceptron(
        bool,
        Topology&& topology,
        bool bias,
        activation_function activation
    ) : structure_(topology, bias),
        activation_(activation)
    {}

    MLPStructure structure_;
    activation_function activation_;
};

}