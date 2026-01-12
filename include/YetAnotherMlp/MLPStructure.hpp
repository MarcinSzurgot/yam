#pragma once

#include "ArrayOfArrays.hpp"
#include "Utils.hpp"

#include <iostream>

namespace yam {

struct MLPStructure {

    template<Integers Topology>
    MLPStructure(
        Topology&& topology,
        bool biases
    ) : MLPStructure(true, topology, biases) { }

    template<std::integral I>
    MLPStructure(
        std::initializer_list<I> topology,
        bool biases
    ) : MLPStructure(true, topology, biases) { }

    auto topology() const -> std::span<const int> { return topology_; }

    auto neurons() const -> std::span<const float> { return neurons_; }
    auto neurons()       -> std::span<      float> { return neurons_; }

    auto weights() const -> std::span<const float> { return weights_; }
    auto weights()       -> std::span<      float> { return weights_; }

    auto biases() const -> std::span<const float> { return biases_; }
    auto biases()       -> std::span<      float> { return biases_; }

private:
    template<Integers Topology>
    MLPStructure(
        bool,
        Topology&& topology,
        bool biases
    ) : topology_(std::begin(topology), std::end(topology)),
        neurons_(sum(topology_)),
        weights_(sum(weightsLayers(topology))),
        biases_(sum(biases ? yam::subrange(1, topology) : yam::empty(topology)))
    {
    }

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
};

}