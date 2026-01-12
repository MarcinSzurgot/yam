#include <iostream>

#include <YetAnotherMlp/MLPerceptron.hpp>
#include <YetAnotherMlp/MLPTrainer.hpp>
#include <YetAnotherMlp/Random.hpp>

#include <gtest/gtest.h>

TEST(TestMLTrainer, testBackpropagation) {
    auto mlp = yam::MLPerceptron({2, 10, 5, 4, 1}, true, yam::Activation::sigmoid);
    auto trainer = yam::MLPTrainer();

    const auto input = std::vector {
        1.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    const auto expected = std::vector {
        1.0f,
        0.0f,
        1.0f,
        0.0f
    };

    const auto error = trainer.train(mlp, 0.2, 0.01, 100000, input, expected, 4);

    auto weights = mlp.structure().weights().values();

    for (auto w = 0u; w < weights.size(); ++w) {
        std::cout << weights[w] << ", ";
    }
    std::cout << "\n";

    for(const auto i : mlp.forward(std::vector {0.0f, 1.0f})) {
        std::cout << i << ", ";
    }
    std::cout << "\n";
    std::cout << error << "\n";
}