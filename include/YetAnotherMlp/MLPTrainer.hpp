#pragma once

#include "Activation.hpp"
#include "Dataset.hpp"
#include "MLPerceptron.hpp"
#include "Random.hpp"
#include "Mathematics.hpp"

#include <algorithm>
#include <ranges>
#include <span>

namespace yam {

struct MLPTrainer {
    MLPTrainer(
        MLPerceptron trainee,
        Dataset trainset,
        Dataset testset,
        derivation_function derivative,
        float learnrate,
        float minError,
        int maxEpochs
    ) : trainee_(std::move(trainee)),
        trainset_(std::move(trainset)),
        testset_(std::move(testset)),
        derivative_(derivative),
        learnrate_(learnrate),
        minError_(minError),
        maxEpochs_(maxEpochs),
        errors_(trainee_.neurons().size()),
        derived_(trainee_.neurons().size()),
        indexes_(trainset_.size())
    {
        std::ranges::copy(std::views::iota(0u, indexes_.size()), indexes_.begin());

        auto rnd = Random();

        rnd.separated(0.3f, 0.2f, trainee.weights());
        rnd.separated(0.3f, 0.2f, trainee.biases());
    }

    auto trainee() -> MLPerceptron& {
        return trainee_;
    }

    auto train() -> float {

        auto achievedError = this->error(testset_, trainee_);

        std::cout << "Initial error: " << achievedError << "\n";

        for (auto i = 0; i < maxEpochs_ && achievedError > minError_; ++i) {
            std::random_shuffle(indexes_.begin(), indexes_.end());

            train(trainee_, trainset_, derivative_, learnrate_);

            achievedError = this->error(testset_, trainee_);
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

    auto error(
        const Dataset& dataset,
        MLPerceptron& mlp
    ) const -> float {
        auto errors = std::views::iota(0, dataset.size()) | std::views::transform([&](auto i) {
            const auto input = dataset.input(i).begin().base();
            const auto expected = dataset.output(i);
            const auto actual = mlp.forward(input).begin().base();
            return yam::distance(expected, actual);
        });

        return std::ranges::fold_left(errors, 0.0f, std::plus<>{}) / dataset.size();
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

    MLPerceptron trainee_;

    Dataset testset_;
    Dataset trainset_;
    
    derivation_function derivative_;

    float learnrate_;
    float minError_;
    int maxEpochs_;

    mutable std::vector<float> errors_;
    mutable std::vector<float> derived_;
    mutable std::vector<int> indexes_;
};

}