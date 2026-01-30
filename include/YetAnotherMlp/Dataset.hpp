#pragma once

#include <span>
#include <vector>

namespace yam {

struct Dataset {
private:
    template<typename Vector>
    auto batch(Vector&& vector, int i) const {
        const auto batchSize = vector.size() / size_;
        return std::span(
            vector.begin() + (i + 0) * batchSize,
            vector.begin() + (i + 1) * batchSize
        );
    }

public:
    Dataset() : Dataset({}, {}, 0) { }

    Dataset(
        std::vector<float> inputs,
        std::vector<float> outputs,
        int size
    ) : inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        size_(size)
    {

    }

    auto size() const -> int { return size_; }
    auto inputSize() const -> int { return size() ? input(0).size() : 0; }
    auto outputSize() const -> int { return size() ? output(0).size() : 0; }

    auto input (int i) const -> std::span<const float> { return batch(inputs_, i); }
    auto input (int i)       -> std::span<      float> { return batch(inputs_, i); }
    auto output(int i) const -> std::span<const float> { return batch(outputs_, i); }
    auto output(int i)       -> std::span<      float> { return batch(outputs_, i); }

private:
    std::vector<float> inputs_;
    std::vector<float> outputs_;
    int size_;
};

}