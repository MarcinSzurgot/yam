#include <algorithm>
#include <functional>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>
#include <ranges>
#include <tuple>
#include <array>
#include <concepts>

double sigmoid(double x) { 
    return 1 / (1 + std::exp(-x)); 
};

double sigmoidDerivative(double x) {
    return x * (1 - x);
}

template<typename T>
concept UnsignedInt = std::is_unsigned_v<T>;

template<typename T>
concept Indexable = requires(T a, typename T::size_type i) {
    { a[i] } -> std::convertible_to<typename T::value_type>;
};

// row * cols + col
// x1 * D0 + x0
// x(n) * D(n - 1) + x(n - 1)
// x0 * D1 * D2 + x1 * D2 + x2
// ((x0 * D1 + x1) * D2 + x2) * D3 ... 

template<typename T, std::size_t D>
struct Matrix {

    template<UnsignedInt... Dimensions>
    Matrix(
        Dimensions&&... dimensions
    ) : dimensions_(dimensions...),
        cells_((dimensions + ...)) {
    }

    // template<UnsignedInt... Indexes>
    // requires requires(Indexes)
    // UnsignedInt auto operator()(Indexes&&... indexes) {
    //     return (dimensions_[indexes] + ...);
    // }

private:
    std::array<T, D> dimensions_;
    std::vector<T> cells_;
};

int main() {
    // auto mat = Matrix<int, 3>(2u, 2u, 2u);
    // auto cell = mat(1u, 2u, 0u);
    // const auto seed = std;
    //     ::chrono
    //     ::high_resolution_clock
    //     ::now()
    //     .time_since_epoch()
    //     .count();

    // auto generator = std::mt19937(seed);

    // const auto input = std::vector<double>{2.0};
    // const auto topology = std::vector{(int) input.size(), 1};
    // const auto bias = false;
    // const auto activation = lm2(std::ranges::transform(x, begin(y), sigmoid));

    // auto summed = createNeurons(topology);
    // auto activated = createNeurons(topology);
    // auto weights = createWeights(topology, bias);

    // std::ranges::generate(weights, lm0(std::uniform_real_distribution(-0.5, 0.5)(generator)));

    // forward(
    //     topology, 
    //     bias, 
    //     activation, 
    //     summed, 
    //     activated, 
    //     weights, 
    //     input
    // );

    // return 0;
}