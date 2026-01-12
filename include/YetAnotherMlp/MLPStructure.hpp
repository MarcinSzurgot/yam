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

    auto topology() const -> std::span<const int> { return neurons_.rows(); }

    auto neurons() const -> const ArrayOfArrays<float>& { return neurons_; }
    auto neurons()       ->       ArrayOfArrays<float>& { return neurons_; }

    auto weights() const -> const ArrayOfArrays<float>& { return weights_; }
    auto weights()       ->       ArrayOfArrays<float>& { return weights_; }

    auto biases() const -> const ArrayOfArrays<float>& { return biases_; }
    auto biases()       ->       ArrayOfArrays<float>& { return biases_; }

private:
    template<Integers Topology>
    MLPStructure(
        bool,
        Topology&& topology,
        bool biases
    ) : neurons_(topology),
        weights_(weightsLayers(topology)),
        biases_(biases ? yam::subrange(1, topology) : yam::empty(topology))
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

    ArrayOfArrays<float> neurons_;
    ArrayOfArrays<float> weights_;
    ArrayOfArrays<float> biases_;
};

}