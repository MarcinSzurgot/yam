#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <span>

namespace yam {

enum class ActivationFunctionType {
    Linear,
    Sigmoid,
    Bisigmoid
};

using activation_function = std::function<float*(std::span<const float>, float*)>;
using derivation_function = std::function<float*(std::span<const float>, float*)>;

template<typename Functor>
auto activation(Functor&& functor) -> activation_function {
    return [functor](std::span<const float> args, float* result) -> float* {
        return std::ranges::transform(args, result, functor).out;
    };
}

struct Activation {
    struct Sigmoid {
        template<std::floating_point F>
        auto operator()(F x) const -> F {
            return 1 / (1 + std::exp(-x));
        }
    };

    struct Bisigmoid {
        template<std::floating_point F>
        auto operator()(F x) const -> F {
            return 2 * Sigmoid()(x) - 1;
        }
    };

    struct Linear {
        template<std::floating_point F>
        auto operator()(F x) const -> F { return x; }
    };

    inline static const auto linear = yam::activation(Linear());
    inline static const auto sigmoid = yam::activation(Sigmoid());
    inline static const auto bisigmoid = yam::activation(Bisigmoid());
};

struct Derivation {
    struct Sigmoid {
        template<std::floating_point F>
        auto operator()(F x) const -> F {
            return x * (1 - x);
        }
    };

    struct Bisigmoid {
        template<std::floating_point F>
        auto operator()(F x) const -> F {
            return Sigmoid()(x) * 2;
        }
    };

    struct Linear {
        template<std::floating_point F>
        auto operator()(F x) const -> F { return 1; }
    };

    inline static const auto linear = yam::activation(Linear());
    inline static const auto sigmoid = yam::activation(Sigmoid());
    inline static const auto bisigmoid = yam::activation(Bisigmoid());
};

}