#pragma once

#include "Utils.hpp"

#include <iostream>
#include <span>
#include <vector>

namespace yam {

template<typename T>
struct ArrayOfArrays {
    using value_type = T;

    template<Integers Rows>
    ArrayOfArrays(Rows&& rows) : ArrayOfArrays(true, rows) { }

    template<std::integral I>
    ArrayOfArrays(std::initializer_list<I> rows) : ArrayOfArrays(true, rows) { }

    ArrayOfArrays() : ArrayOfArrays(std::initializer_list<int> {}) { }

    auto values() const -> std::span<const value_type> { return values_; }
    auto values()       -> std::span<      value_type> { return values_; }
    auto rows() const -> std::span<const int> { return rows_; }

    auto row(int row) const -> std::span<const value_type> { return yam::row(values_, rows_, row); }
    auto row(int row)       -> std::span<      value_type> { return yam::row(values_, rows_, row); }

private:
    template<Integers Rows>
    ArrayOfArrays(
        bool,
        Rows&& rows
    ) : rows_(std::begin(rows), std::end(rows)),
        values_(yam::sum(rows))
    {

    }

    std::vector<int> rows_;
    std::vector<value_type> values_;
};

};