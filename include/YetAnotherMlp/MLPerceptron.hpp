#pragma once

#include "Activation.hpp"
#include "Mathematics.hpp"
#include "Utils.hpp"

#include <functional>

namespace yam {

struct MLPerceptron {

    using activation_function = std::function<float*(std::span<const float>, float*)>;

    MLPerceptron() : MLPerceptron({1, 1}, false, yam::Activation::linear) { }

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

    auto forward(const float* input) -> std::span<const float> {
        auto lower = input;
        auto upper = neurons_.begin().base();
        auto last  = neurons_.end().base();
        auto weight = weights_.begin();
        auto bias = biases_.begin().base();

        std::fill(upper, last, 0);

        for (const auto [lc, uc] : std::views::adjacent<2>(topology_)) {
            // for (auto u = 0; u < uc; ++u) {
            //     for (auto l = 0; l < lc; ++l) {
            //         upper[u] += *weight++ * lower[l];
            //     }
            // }

            matmul(
                std::span(weight, weight + uc * lc),
                std::span(lower, lower + lc),
                upper,
                lc
            );

            for (auto u = bias ? 0 : uc; u < uc; ++u) {
                upper[u] += *bias++;
            }

            lower = upper;
            upper = activation_({upper, upper + uc}, upper);
        }

        return {lower, last};
    }

    auto topology() const -> std::span<const int> { return topology_; }

    auto neurons() const -> std::span<const float> { return neurons_; }
    auto neurons()       -> std::span<      float> { return neurons_; }

    auto weights() const -> std::span<const float> { return weights_; }
    auto weights()       -> std::span<      float> { return weights_; }

    auto biases() const -> std::span<const float> { return biases_; }
    auto biases()       -> std::span<      float> { return biases_; }

private:
    template<Integers Topology>
    MLPerceptron(
        bool,
        Topology&& topology,
        bool bias,
        activation_function activation
    ) : topology_(std::begin(topology), std::end(topology)),
        activation_(activation)
    {
        const auto [neurons, weights] = std::ranges::fold_left(
            std::views::adjacent<2>(topology_),
            std::pair<int, int>(),
            [](auto&& r, auto&& c) {
                return std::make_pair(
                    r.first  + c.second, // summing neurons of upper layers
                    r.second + c.second * c.first // multiplicating neurons of adjacent layers
                );
            }
        );

        neurons_.resize(neurons);
        biases_.resize(bias ? neurons : 0u);
        weights_.resize(weights);
    }

    std::vector<int> topology_;
    std::vector<float> neurons_;
    std::vector<float> weights_;
    std::vector<float> biases_;
    activation_function activation_;
};

}