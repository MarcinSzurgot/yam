#pragma once

#include <concepts>
#include <iterator>
#include <ranges>

namespace yam {

struct Square {
    template<std::floating_point F>
    auto operator()(F lhs, F rhs) {
        const auto diff = lhs - rhs;
        return diff * diff;
    }
};

template<
    std::ranges::forward_range Range,
    std::forward_iterator Iter
> requires (
    std::floating_point<std::ranges::range_value_t<Range>> 
    && std::floating_point<std::iter_value_t<Iter>>
)
auto distance(
    Range&& from,
    Iter to
) -> std::ranges::range_value_t<Range> {
    return std::transform_reduce(
        std::begin(from),
        std::end(from),
        to,
        std::ranges::range_value_t<Range>(),
        std::plus<>{},
        yam::Square()
    );
}

}