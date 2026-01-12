#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <numeric>
#include <ranges>
#include <span>

namespace yam {

template<typename T>
concept Integers = std::ranges::input_range<T> && std::integral<std::ranges::range_value_t<T>>;

template<
    std::ranges::input_range Numbers,
    typename V = std::ranges::range_value_t<Numbers>
>
requires (std::is_arithmetic_v<V>)
auto sum(Numbers&& numbers) -> V {
    return std::ranges::fold_left(numbers, V(), std::plus<>{});
}

template<std::ranges::forward_range Range>
auto subrange(Range&& range, int size) {
    const auto first = std::begin(range);
    return std::ranges::subrange(first, std::next(first, size));
}

template<std::ranges::forward_range Range>
auto subrange(int offset, Range&& range) {
    const auto first = std::begin(range);
    const auto last  = std::end(range);
    const auto populated = first != last;

    return std::ranges::subrange(populated ? std::next(first) : first, last);
}

template<std::ranges::forward_range Range>
auto empty(Range&& range) {
    return std::ranges::subrange(std::begin(range), std::begin(range));
}

template<
    std::ranges::contiguous_range Range,
    Integers Rows,
    typename I = std::ranges::range_value_t<Rows>
>
auto row(Range&& range, Rows&& rows, I row) {
    const auto first = std::begin(rows);
    const auto offset = sum(subrange(rows, row));
    return std::span(
        std::begin(range) + offset,
        std::begin(range) + offset + *(first + row)
    );
}

template<std::ranges::bidirectional_range Range>
auto adjacent(Range&& range) {
    const auto first = std::begin(range);
    const auto last  = std::end(range);
    const auto populated = first != last;
    return std::views::zip(
        std::ranges::subrange(first, populated ? std::prev(last) : last),
        std::ranges::subrange(populated ? std::next(first) : first, last)
    );
}

}