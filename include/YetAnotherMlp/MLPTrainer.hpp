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
    ) -> float {
        init(dataset, trainee);

        auto achievedError = std::numeric_limits<float>::max();

        for (auto i = 0; i < maxEpochs && achievedError > error; ++i) {

            std::random_shuffle(indexes_.begin(), indexes_.end());

            train(trainee, dataset, derivation, learnrate);

            achievedError = this->error(dataset, trainee);
        }

        return achievedError;
    }

private:
    auto train(
        MLPerceptron& trainee, 
        const Dataset& dataset,
        derivation_function derivation,
        float learnrate
    ) -> void {
        for(const auto i : indexes_) {
            const auto input = dataset.input(i).begin().base();
            const auto expected = dataset.output(i).begin().base();
            backpropagate(trainee, learnrate, input, expected, derivation);
        }
    }

    auto init(const Dataset& dataset, MLPerceptron& trainee) const -> void {
        errors_.resize(trainee.neurons().size() - trainee.topology()[0]);
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
    ) const {
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
        float* error,
        float* weight,
        float* bias,
        float* signal,
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