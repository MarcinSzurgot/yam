#include <YetAnotherMlp/Activation.hpp>
#include <YetAnotherMlp/MLPerceptron.hpp>

#include <gtest/gtest.h>

#include <cmath>

TEST(TestMLPerceptron, SingleWeightLinearActivation) {
    auto mlp = yam::MLPerceptron({1, 1}, false, yam::Activation::linear);

    mlp.structure().weights().values()[0] = 3;

    const auto input = std::vector { 5.0f };
    const auto result = mlp.forward(input);
    
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0], 15);
}

TEST(TestMLPerceptron, SingleWeightSigActivation) {
    auto mlp = yam::MLPerceptron({1, 1}, false, yam::Activation::sigmoid);

    mlp.structure().weights().values()[0] = 1.5;

    const auto input = std::vector { -1.0f };
    const auto result = mlp.forward(input);
    
    ASSERT_EQ(result.size(), 1);
    ASSERT_FLOAT_EQ(result[0], 0.1824255238f);
}

TEST(TestMLPerceptron, DoubleWeightLinearActivation) {
    auto mlp = yam::MLPerceptron({2, 1}, false, yam::Activation::linear);

    mlp.structure().weights().row(0)[0] = 1.5;
    mlp.structure().weights().row(0)[1] = 3.7;

    const auto input = std::vector { -1.0f, 2.0f };
    const auto result = mlp.forward(input);
    
    ASSERT_EQ(result.size(), 1);
    ASSERT_FLOAT_EQ(result[0], 5.9f);
}

TEST(TestMLPerceptron, DoubleLayersLinearActivation) {
    auto mlp = yam::MLPerceptron({2, 2, 1}, true, yam::Activation::linear);

    mlp.structure().weights().row(0)[0] = 1;
    mlp.structure().weights().row(0)[2] = 3;
    mlp.structure().biases().row(0)[0] = 1.5;

    mlp.structure().weights().row(0)[1] = 2;
    mlp.structure().weights().row(0)[3] = 4;
    mlp.structure().biases().row(0)[1] = 1.2;

    mlp.structure().weights().row(1)[0] = 5;
    mlp.structure().weights().row(1)[1] = 6;
    mlp.structure().biases().row(1)[0] = 2.3;

    const auto input = std::vector { -0.3f, 0.6f };
    const auto result = mlp.forward(input);
    
    ASSERT_EQ(result.size(), 1);
    ASSERT_FLOAT_EQ(result[0], 35.3f);
}