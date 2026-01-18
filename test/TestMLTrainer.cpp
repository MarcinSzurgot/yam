#include <iostream>

#include <YetAnotherMlp/Dataset.hpp>
#include <YetAnotherMlp/MLPerceptron.hpp>
#include <YetAnotherMlp/MLPTrainer.hpp>
#include <YetAnotherMlp/Random.hpp>

#include <gtest/gtest.h>



TEST(TestMLTrainer, testBackpropagation) {
    auto mlp = yam::MLPerceptron({2, 5, 1}, true, yam::Activation::sigmoid);
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

    const auto dataset = yam::Dataset(input, expected, 4);

    const auto error = trainer.train(mlp, 0.5, 0.01, 10000, dataset, yam::Derivation::sigmoid);

    for (const auto w : mlp.weights()) {
        std::cout << w << ", ";
    }
    std::cout << "\n";

    for(const auto i : mlp.forward(&input[2 * 2])) {
        std::cout << i << ", ";
    }
    std::cout << "\n";
    std::cout << error << "\n";
}