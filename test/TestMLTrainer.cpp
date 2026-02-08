#include <iostream>

#include <YetAnotherMlp/Dataset.hpp>
#include <YetAnotherMlp/MLPerceptron.hpp>
#include <YetAnotherMlp/MLPTrainer.hpp>
#include <YetAnotherMlp/Mnist.hpp>
#include <YetAnotherMlp/Random.hpp>

#include <gtest/gtest.h>

TEST(TestMLTrainer, learningXor) {
    const auto mlp = yam::MLPerceptron({2, 2, 1}, true, yam::Activation::sigmoid);

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

    auto trainer = yam::MLPTrainer(
        mlp, dataset, dataset, yam::Derivation::sigmoid, 20, 0.01, 4000
    );

    auto actualError = std::numeric_limits<float>::max();
    for (auto i = 0; i < 10 && actualError > expectedError; ++i) {
        actualError = trainer.train();
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

    const auto mlp = yam::MLPerceptron({1, 20, 10, 5, 1}, true, yam::Activation::sigmoid);

    auto trainer = yam::MLPTrainer(
        mlp, dataset, dataset, yam::Derivation::sigmoid, 0.05, 0.001, 100000
    );

    auto actualError = trainer.train();

    for (const auto x : inputs) {
        const auto result = trainer.trainee().forward(&x)[0];

        std::cout << "sin(" << x * 360 << ") = " << result * 2.4 - 1.2 << "\n";
    }

    std::cout << actualError << "\n";
}

TEST(TestMLTrainer, learningMnist) {
    const auto trainset = yam::Mnist::read(
        "../resources/train-images.idx3-ubyte", 
        "../resources/train-labels.idx1-ubyte"
    );

    const auto testset = yam::Mnist::read(
        "../resources/t10k-images.idx3-ubyte", 
        "../resources/t10k-labels.idx1-ubyte"
    );
    
    auto mlp = yam::MLPerceptron(
        {trainset.inputSize(), 100, trainset.outputSize()}, 
        true, 
        yam::Activation::sigmoid
    );

    auto trainer = yam::MLPTrainer(
        mlp, trainset, testset, yam::Derivation::sigmoid, 0.1, 0.07, 1000
    );

    const auto error = trainer.train();

    for (auto i = 0; i < testset.size(); ++i) {
        const auto result = mlp.forward(testset.input(i).begin().base());
        const auto output = testset.output(i);

        const auto actual = result.end() - std::ranges::max_element(result) - 1;
        const auto expected = output.end() - std::ranges::max_element(output) - 1;

        std::cout << actual << ", " << expected << "\n";
    }

    std::cout << "Mnist error: " << error << "\n";
}