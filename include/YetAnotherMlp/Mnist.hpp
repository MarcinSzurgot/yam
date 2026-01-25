#pragma once

#include "Dataset.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <optional>
#include <string_view>
#include <vector>

namespace yam {

struct Mnist {

struct ImageStructure {
    std::uint32_t pictures;
    std::uint32_t imageHeight;
    std::uint32_t imageWidth;
};

struct LabelStructure {
    std::uint32_t labels;
};

static auto read(
    const char* imagesFilename,
    const char* labelsFilename
) -> Dataset {
    auto images = std::ifstream(imagesFilename, std::ios::binary | std::ios::ate);

    const auto size = images.tellg();

    auto buffer = std::vector<unsigned char>(size);
    images.seekg(0, std::ios::beg);
    images.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    const auto mnist = mnistStructure(buffer);
    if (mnist == std::nullopt) {
        return Dataset();
    }

    const auto [structure, next] = *mnist;

    auto inputs = std::vector<float>(
        structure.pictures 
        * structure.imageHeight 
        * structure.imageWidth
    );

    std::transform(next, buffer.end(), inputs.begin(), [](std::uint8_t c) {
        return c / 256.0f;
    });

    auto labels = std::ifstream(labelsFilename, std::ios::binary | std::ios::ate);

    buffer.resize(labels.tellg());

    labels.seekg(0, std::ios::beg);
    labels.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    const auto [magicNumber, next2] = fromLittleEndian<std::uint32_t>(buffer.begin(), buffer.end());
    const auto [labelsNumber, next3] = fromLittleEndian<std::uint32_t>(next2, buffer.end());

    auto outputs = std::vector<float>(labelsNumber * 10, 0);
    for (auto i = 0u; i < labelsNumber; ++i) {
        outputs[i * 10 + next3[i]] = 1;
    }

    return Dataset(
        inputs, 
        outputs,
        labelsNumber
    );
}

template<std::integral I, std::input_iterator Iterator>
requires (
    std::integral<std::iter_value_t<Iterator>>
    && sizeof(std::iter_value_t<Iterator>) == 1
)
static auto fromLittleEndian(
    Iterator first, 
    Iterator last
) -> std::pair<I, Iterator> {
    const auto size = sizeof(I);

    auto value = I();

    for (auto i = 0u; i < size && first != last; ++i) {
        value = (value << sizeof(std::uint8_t) * 8) | *first++;
    }

    return std::make_pair(value, first);
}

template<std::ranges::input_range Range>
requires (
    std::integral<std::ranges::range_value_t<Range>>
    && sizeof(std::ranges::range_value_t<Range>) == 1
)
static auto mnistStructure(
    Range&& range
) -> std::optional<std::pair<ImageStructure, std::ranges::iterator_t<Range>>> {
    auto first = std::begin(range);
    auto last  = std::end(range);
    const auto [magicNumber, next] = fromLittleEndian<std::uint32_t>(first, last);

    if (magicNumber != 2051) {
        return std::nullopt;
    }

    const auto [pictures, next2] = fromLittleEndian<std::uint32_t>(next, last);
    const auto [imageHeight, next3] = fromLittleEndian<std::uint32_t>(next2, last);
    const auto [imageWidth, next4] = fromLittleEndian<std::uint32_t>(next3, last);

    return std::make_pair(
        ImageStructure { 
            .pictures = pictures, 
            .imageHeight = imageHeight, 
            .imageWidth = imageWidth
        },
        next4
    );
}



};

}