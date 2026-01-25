#include <iostream>

#include <YetAnotherMlp/Dataset.hpp>
#include <YetAnotherMlp/MLPerceptron.hpp>
#include <YetAnotherMlp/MLPTrainer.hpp>
#include <YetAnotherMlp/Mnist.hpp>
#include <YetAnotherMlp/Random.hpp>

#include <gtest/gtest.h>

TEST(TestMLTrainer, learningXor) {
    auto mlp = yam::MLPerceptron({2, 2, 1}, true, yam::Activation::sigmoid);
    const auto trainer = yam::MLPTrainer();

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

    const auto expectedError = 0.01;
    auto actualError = std::numeric_limits<float>::max();
    for (auto i = 0; i < 10 && actualError > expectedError; ++i) {
        actualError = trainer.train(mlp, 20, 0.01, 4000, dataset, yam::Derivation::sigmoid);
    }

    ASSERT_LE(actualError, expectedError);
}

TEST(TestMLTrainer, learningSinus) {
    auto inputs = std::vector<float>(100);
    auto outputs = std::vector<float>(100);

    for (auto i = 0u; i < inputs.size(); ++i) {
        inputs[i] = (float) i / inputs.size();

        const auto deg = inputs[i] * 2 * M_PI;
        
        outputs[i] = (std::sin(deg) + 1.2) / 2.4;
    }

    const auto dataset = yam::Dataset(inputs, outputs, 100);

    auto mlp = yam::MLPerceptron({1, 20, 10, 5, 1}, true, yam::Activation::sigmoid);

    const auto trainer = yam::MLPTrainer();

    auto actualError = trainer.train(mlp, 0.05, 0.001, 100000, dataset, yam::Derivation::sigmoid);

    for (const auto x : inputs) {
        const auto result = mlp.forward(&x)[0];

        std::cout << "sin(" << x * 360 << ") = " << result * 2.4 - 1.2 << "\n";
    }

    std::cout << actualError << "\n";
}

TEST(TestMLTrainer, learningMnist) {
    const auto dataset = yam::Mnist::read(
        "../resources/train-images.idx3-ubyte", 
        "../resources/train-labels.idx1-ubyte"
    );
    
    auto mlp = yam::MLPerceptron({28 * 28, 800, 10}, true, yam::Activation::sigmoid);

    const auto trainer = yam::MLPTrainer();

    const auto error = trainer.train(mlp, 0.1, 0.02, 1000, dataset, yam::Derivation::sigmoid);

    std::cout << "Mnist error: " << error << "\n";
}