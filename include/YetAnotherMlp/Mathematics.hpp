#pragma once

#include <concepts>
#include <iterator>
#include <numeric>
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

// multiplier is a transposed matrix
template<
    std::ranges::forward_range Multiplicand,
    std::ranges::forward_range Multiplier,
    std::input_or_output_iterator Multiplication,
    typename Operator,
    typename Reductor
>
auto matmul(
    Multiplicand&& multiplicand,
    Multiplier&& mutliplier, // transposed matrix
    Multiplication multiplication,
    std::size_t common,
    Operator&& op,
    Reductor&& rd
) -> Multiplication {
    auto m1 = std::begin(multiplicand);
    auto m1Last = std::end(multiplicand);
    auto m2Last = std::end(mutliplier);

    for (; m1 != m1Last;) {
        auto m1first = m1;
        auto m2 = std::begin(mutliplier);

        for (; m2 != m2Last;) {
            m1 = m1first;

            auto sum = decltype(op(*m1++, *m2++))();
            for (auto c = 0u; c < common; ++c) {
                sum = rd(sum, op(*m1++, *m2++));
            }
            *multiplication++ = sum;
        }
    }

    return multiplication;
}

template<
    std::ranges::forward_range Multiplicand,
    std::ranges::forward_range Multiplier,
    std::input_or_output_iterator Multiplication
>
auto matmul(
    Multiplicand&& multiplicand,
    Multiplier&& mutliplier,
    Multiplication multiplication,
    std::size_t common
) -> Multiplication {
    return matmul(
        std::forward<Multiplicand>(multiplicand),
        std::forward<Multiplier>(mutliplier),
        multiplication,
        common,
        std::multiplies<>{},
        std::plus<>{}
    );
}

}