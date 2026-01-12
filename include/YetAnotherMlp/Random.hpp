#pragma once

#include <algorithm>
#include <chrono>
#include <concepts>
#include <random>
#include <ranges>

namespace yam {

struct Random {
    Random(std::uint32_t seed) : generator_(seed) { } 
    Random() : Random(now().time_since_epoch().count()) { }

    static auto now() -> std::chrono::high_resolution_clock::time_point {
        return std::chrono::high_resolution_clock::now();
    }

    template<std::integral I>
    auto operator()(I min, I max) -> I {
        return std::uniform_int_distribution(min, max)(generator_);
    }

    template<std::floating_point F>
    auto operator()(F min, F max) -> F {
        return std::uniform_real_distribution(min, max)(generator_);
    }

    template<std::ranges::forward_range Range, typename A>
    requires (std::is_arithmetic_v<A>)
    void operator()(A min, A max, Range&& range) { 
        std::ranges::generate(range, [this, min, max] { return (*this)(min, max); }); 
    }

    template<typename A>
    requires (std::is_arithmetic_v<A>)
    auto separated(A bigRadius, A smallRadius) -> A {
        const auto value = (*this)(-smallRadius, smallRadius);
        return value + (value < 0 ? -1 : 1) * bigRadius;
    }

    template<std::ranges::forward_range Range, typename A>
    requires (std::is_arithmetic_v<A>)
    void separated(A bigRadius, A smallRadius, Range&& range) {
        std::ranges::generate(range, [this, bigRadius, smallRadius] { 
            return this->separated(bigRadius, smallRadius);
        });
    }

private:
    std::mt19937 generator_;
};

}