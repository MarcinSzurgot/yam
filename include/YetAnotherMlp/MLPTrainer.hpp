#pragma once

#include "Activation.hpp"
#include "Dataset.hpp"
#include "MLPerceptron.hpp"
#include "Random.hpp"

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
        const Dataset& dataset,
        derivation_function derivation
    ) const -> float {
        init(dataset, trainee);

        auto achievedError = this->error(dataset, trainee);

        std::cout << "Initial error: " << achievedError << "\n";

        for (auto i = 0; i < maxEpochs && achievedError > error; ++i) {
            std::random_shuffle(indexes_.begin(), indexes_.end());

            train(trainee, dataset, derivation, learnrate);

            achievedError = this->error(dataset, trainee);
            std::cout << "Epoch: " << i << ", error: " << achievedError << "\n";
        }

        return achievedError;
    }

private:
    auto train(
        MLPerceptron& trainee, 
        const Dataset& dataset,
        derivation_function derivation,
        float learnrate
    ) const -> void {
        for(const auto i : indexes_) {
            const auto input = dataset.input(i).begin().base();
            const auto expected = dataset.output(i).begin().base();
            backpropagate(trainee, learnrate, input, expected, derivation);
        }
    }

    auto init(const Dataset& dataset, MLPerceptron& trainee) const -> void {
        errors_.resize(trainee.neurons().size());
        derived_.resize(errors_.size());
        indexes_.resize(dataset.size());

        for (auto i = 0u; i < indexes_.size(); ++i) {
            indexes_[i] = i;
        }

        auto rnd = Random();

        rnd.separated(-0.3f, 0.2f, trainee.weights());
        rnd.separated(-0.3f, 0.2f, trainee.biases());
    }

    auto error(
        const Dataset& dataset,
        MLPerceptron& mlp
    ) const -> float {
        float error = 0.0f;
        for(const auto i : indexes_) {
            const auto input = dataset.input(i).begin().base();
            const auto expected = dataset.output(i);
            const auto actual = mlp.forward(input);
            error += mse(expected, actual);
        }
        return error / indexes_.size();
    }

    auto mse(
        std::span<const float> expected,
        std::span<const float> actual
    ) const -> float {
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
    ) const {
        const auto actual = trainee.forward(input).begin().base();
        const auto topology = trainee.topology();

        derivative(trainee.neurons(), derived_.begin().base());

        auto error = errors_.end().base() - topology.back();
        auto derived = derived_.end().base() - topology.back();
        const auto* signal = trainee.neurons().end().base() - topology.back();
        auto weight = trainee.weights().end().base();
        auto bias = trainee.biases().end().base();

        lastLayerError(error, derived, actual, expected, topology.back());

        for (auto layer = topology.size(); layer > 1u; --layer) {
            const auto uc = topology[layer - 1];
            const auto lc = topology[layer - 2];

            error -= lc;
            derived -= lc;
            signal = layer > 2u ? signal - lc : input;
            weight -= lc * uc;
            bias -= bias ? uc : 0u;

            if (layer > 2u) {
                hiddenLayerError(error, derived, weight, lc, uc);
            }
            correct(error + lc, weight, bias, signal, learnrate, lc, uc);
        }
    }

    void lastLayerError(
        float* error,
        const float* derived,
        const float* actual,
        const float* expected,
        int size
    ) const {
        for (auto i = 0u; i < size; ++i) {
            error[i] = derived[i] * (expected[i] - actual[i]);
        }
    }

    void hiddenLayerError(
              float* error,
        const float* derived,
        const float* weight,
        int lc,
        int uc
    ) const {
        for (auto l = 0u; l < lc; ++l) {
            error[l] = 0;
            for(auto u = 0u; u < uc; ++u) {
                error[l] += *weight++ * error[lc + u];
            }
            error[l] *= derived[l];
        }
    }

    void correct(
        const float* error,
              float* weight,
              float* bias,
        const float* signal,
        float learnrate,
        int lc,
        int uc
    ) const {
        for (auto l = 0u; l < lc; ++l) {
            for (auto u = 0u; u < uc; ++u) {
                *weight++ += learnrate * error[u] * signal[l];
            }
        }

        for (auto b = bias ? 0u : uc; b < uc; ++b) {
            bias[b] += learnrate * error[b];
        }
    }

    mutable std::vector<float> errors_;
    mutable std::vector<float> derived_;
    mutable std::vector<int> indexes_;
};

}