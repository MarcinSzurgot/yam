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
        auto lower = neurons_.begin().base();
        auto upper = std::ranges::copy(input, lower).out;
        auto last  = neurons_.end().base();
        auto weight = weights_.begin();
        auto bias = biases_.begin().base();

        std::fill(upper, last, 0);

        for (const auto [lc, uc] : adjacent(topology_)) {
            for (auto l = 0; l < lc; ++l) {
                for (auto u = 0; u < uc; ++u) {
                    upper[u] += *weight++ * lower[l];
                }
            }

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
        neurons_(sum(topology_)),
        weights_(sum(weightsLayers(topology))),
        biases_(bias ? sum(topology_) - topology_[0] : 0),
        activation_(activation)
    {}

    template<Integers Topology>
    auto weightsLayers(Topology&& topology) {
        auto layersSizes = std::vector<int>();
        for (const auto [lower, upper] : adjacent(topology)) {
            layersSizes.push_back(lower * upper);
        }
        return layersSizes;
    }

    std::vector<int> topology_;
    std::vector<float> neurons_;
    std::vector<float> weights_;
    std::vector<float> biases_;
    activation_function activation_;
};

}