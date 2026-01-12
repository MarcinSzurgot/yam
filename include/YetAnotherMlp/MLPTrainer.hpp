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
        std::span<const float> outputs
    ) -> float {
        errors_.resize(trainee.neurons().size());
        derived_.resize(errors_.size() - trainee.topology()[0]);

        auto rnd = Random();

        rnd.separated(-0.3f, 0.2f, trainee.weights());
        rnd.separated(-0.3f, 0.2f, trainee.biases());

        auto achievedError = 9999999.0f;

        const auto inputSize = trainee.topology().front();
        const auto outputSize = trainee.topology().back();
        const auto samples = inputs.size() / inputSize;

        auto indexes = std::vector<int>(samples);
        for (auto i = 0u; i < indexes.size(); ++i) {
            indexes[i] = i;
        }

        for (auto i = 0; i < maxEpochs && achievedError > error; ++i) {

            std::random_shuffle(indexes.begin(), indexes.end());

            for (auto s = 0; s < samples; ++s) {
                const auto input = &inputs[indexes[s] * inputSize];
                const auto expected = &outputs[indexes[s] * outputSize];
                backpropagate(trainee, learnrate, input, expected, Derivation::sigmoid);
            }

            achievedError = 0.0f;
            for (auto s = 0; s < samples; ++s) {
                const auto input = &inputs[indexes[s] * inputSize];
                const auto expected = &outputs[indexes[s] * outputSize];
                const auto actual = trainee.forward(input);
                achievedError += mse({expected, expected + actual.size()}, actual);
            }
            achievedError /= samples;
        }

        return achievedError;
    }

private:
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
        const float* input,
        const float* expected,
        derivation_function derivative
    ) {
        const auto actual = trainee.forward(input).begin().base();
        const auto topology = trainee.topology();

        derivative(trainee.neurons().subspan(topology[0]), derived_.begin().base());

        auto error = errors_.end().base() - topology.back();
        auto derived = derived_.end().base() - topology.back();
        auto signal = trainee.neurons().end().base() - topology.back();
        auto weight = trainee.weights().end().base();
        auto bias = trainee.biases().end().base();

        lastLayerError(error, derived, actual, expected, topology.back());

        for (auto layer = topology.size() - 1; layer > 1u; --layer) {
            const auto uc = topology[layer];
            const auto lc = topology[layer - 1];

            error -= lc;
            derived -= lc;
            signal -= lc;
            weight -= lc * uc;
            bias -= bias ? uc : 0u;

            hiddenLayerError(error, derived, weight, lc, uc);
            correct(error + lc, weight, bias, signal, learnrate, lc, uc);
        }

        derived -= topology[0];
        signal -= topology[0];
        weight -= topology[0] * topology[1];
        bias -= bias ? topology[1] : 0u;

        correct(error, weight, bias, signal, learnrate, topology[0], topology[1]);
    }

    void lastLayerError(
        float* error,
        const float* derived,
        const float* actual,
        const float* expected,
        int size
    ) {
        for (auto i = 0u; i < size; ++i) {
            error[i] = derived[i] * (expected[i] - actual[i]);
        }
    }

    void hiddenLayerError(
        float* error,
        float* derived,
        float* weight,
        int lc,
        int uc
    ) {
        for (auto l = 0u; l < lc; ++l) {
            error[l] = 0;
            for(auto u = 0u; u < uc; ++u) {
                error[l] += *weight++ * error[lc + u];
            }
            error[l] *= derived[l];
        }
    }

    void correct(
        float* error,
        float* weight,
        float* bias,
        float* signal,
        float learnrate,
        int lc,
        int uc
    ) {
        for (auto l = 0u; l < lc; ++l) {
            for (auto u = 0u; u < uc; ++u) {
                *weight++ += learnrate * error[u] * signal[l];
            }
        }

        for (auto b = bias ? 0u : uc; b < uc; ++b) {
            bias[b] += learnrate * error[b];
        }
    }

    std::vector<float> errors_;
    std::vector<float> derived_;
};

}