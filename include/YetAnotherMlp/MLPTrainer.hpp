#pragma once

#include "Activation.hpp"
#include "MLPerceptron.hpp"
#include <Random.hpp>

#include <algorithm>
#include <ranges>
#include <span>

namespace yam {

struct MLPTrainer {
    MLPTrainer() {}

    auto train(
        MLPerceptron& trainee,
        float learnrate,
        float error,
        int maxEpochs,
        std::span<const float> inputs,
        std::span<const float> outputs,
        int samples
    ) -> float {
        errors_ = ArrayOfArrays<float>(trainee.structure().topology());
        derived_ = ArrayOfArrays<float>(trainee.structure().topology().subspan(1));

        auto rnd = Random();

        rnd.separated(0.3f, 0.2f, trainee.structure().weights().values());
        rnd.separated(0.3f, 0.2f, trainee.structure().biases().values());

        auto achievedError = 9999999.0f;

        const auto inputSize = inputs.size() / samples;
        const auto outputSize = outputs.size() / samples;

        auto indexes = std::vector<int>(samples);
        for (auto i = 0u; i < indexes.size(); ++i) {
            indexes[i] = i;
        }

        for (auto i = 0; i < maxEpochs && achievedError > error; ++i) {

            std::random_shuffle(indexes.begin(), indexes.end());

            for (auto s = 0; s < samples; ++s) {
                const auto input = inputs.subspan(indexes[s] * inputSize, inputSize);
                const auto expected = outputs.subspan(indexes[s] * outputSize, outputSize);
                backpropagate(trainee, learnrate, input, expected, Derivation::sigmoid);
            }

            achievedError = 0.0f;
            for (auto s = 0; s < samples; ++s) {
                const auto input = inputs.subspan(s * inputSize, inputSize);
                const auto expected = outputs.subspan(s * outputSize, outputSize);
                const auto actual = trainee.forward(input);
                achievedError += mse(expected, actual);
            }
            achievedError /= samples;
        }

        return achievedError;
    }

    auto mse(
        std::span<const float> expected,
        std::span<const float> actual
    ) -> float {
        auto error = 0.0f;
        for (auto i = 0u; i < expected.size(); ++i) {
            error += (expected[i] - actual[i]) * (expected[i] - actual[i]);
        }
        return std::sqrt(error / expected.size());
    }
    
    void backpropagate(
        MLPerceptron& trainee,
        float learnrate,
        std::span<const float> input,
        std::span<const float> expected,
        derivation_function derivative
    ) {
        const auto actual = trainee.forward(input);

        derive(trainee.structure().neurons(), derivative);
        const auto layers = trainee.structure().topology().size();

        lastLayerError(expected, actual);

        for (auto layer = layers - 1; layer > 1u; --layer) {
            hiddenLayerError(trainee, layer - 1);
            correct(trainee, learnrate, layer - 1);
        }

        correct(trainee, learnrate, 0);
    }

    void lastLayerError(
        std::span<const float> expected,
        std::span<const float> actual
    ) {
        auto error = errors_.row(errors_.rows().size() - 1);
        auto derived = derived_.row(derived_.rows().size() - 1);
        for (auto i = 0u; i < expected.size(); ++i) {
            error[i] = derived[i] * (expected[i] - actual[i]);
        }
    }

    void hiddenLayerError(MLPerceptron& trainee, int layer) {
        auto lower = errors_.row(layer);
        auto upper = errors_.row(layer + 1);
        auto derived = derived_.row(layer -1);
        auto weight = trainee.structure().weights().row(layer).begin();

        for (auto l = 0u; l < lower.size(); ++l) {
            lower[l] = 0;
            for (auto u = 0u; u < upper.size(); ++u) {
                lower[l] += *weight++ * upper[u];
            }

            lower[l] *= derived[l];
        }
    }

    void correct(MLPerceptron& trainee, float learnrate, int layer) {
        auto weight = trainee.structure().weights().row(layer).begin();
        auto errors = errors_.row(layer + 1);
        auto signal = trainee.structure().neurons().row(layer);

        for (auto l = 0u; l < signal.size(); ++l) {
            for (auto u = 0u; u < errors.size(); ++u) {
                *weight++ += learnrate * errors[u] * signal[l];
            }
        }

        if (trainee.structure().biases().rows().size() > 0u) {
            auto biases = trainee.structure().biases().row(layer);
            for (auto b = 0u; b < biases.size(); ++b) {
                biases[b] += learnrate * errors[b];
            }
        }
    }

    void derive(
        ArrayOfArrays<float>& neurons,
        derivation_function derivation
    ) {
        auto first = neurons.row(1).begin().base();
        auto last  = neurons.values().end().base();

        derivation({first, last}, derived_.values().begin().base());
    }

private:
    ArrayOfArrays<float> errors_;
    ArrayOfArrays<float> derived_;
};

}