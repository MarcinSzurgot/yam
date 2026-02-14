#pragma once

#include "Activation.hpp"
#include "Algorit.hpp"
#include "Dataset.hpp"
#include "MLPerceptron.hpp"
#include "Random.hpp"
#include "Mathematics.hpp"

#include <algorithm>
#include <ranges>
#include <span>

namespace yam {

struct MLPTrainer {
    MLPTrainer() {}

    MLPTrainer(
        MLPerceptron trainee,
        float learnrate,
        float error,
        int maxEpochs,
        const Dataset& trainset,
        const Dataset& testset,
        derivation_function derivation
    ) : trainee_(std::move(trainee)),
        learnrate_(learnrate),
        error_(error),
        maxEpochs_(maxEpochs),
        trainset_(trainset),
        testset_(testset),
        derivation_(derivation)
    {

    }

    struct Result {
        float targetError;
        int maxEpochs;

        float error;
        int epoch;
        MLPerceptron* trainee;

        auto isTrained() const {
            return error <= targetError || epoch >= maxEpochs;
        }

        friend auto operator==(
            const Result& lhs,
            const Result& rhs
        ) -> bool {
            return lhs.epoch == rhs.epoch && lhs.error == rhs.error;
        }
    };

    auto begin() {
        init(trainset_, trainee_);

        auto initial = Result {
            .targetError = this->error_,
            .maxEpochs = this->maxEpochs_,

            .error = this->error(testset_, trainee_),
            .epoch = 0,
            .trainee = std::addressof(trainee_)
        };

        return Algorit(
            initial,
            [this](auto&& i) { return true; },
            [this](auto&& i) { 
                std::random_shuffle(indexes_.begin(), indexes_.end());

                train(trainee_, trainset_, derivation_, learnrate_);

                i.error = this->error(testset_, trainee_);
                i.epoch++;
            }
        );
    }

    auto end() {
        return std::default_sentinel;
    }

    auto train() -> Result { return train([](auto&&){}); }

    template<typename OnEpoch>
    auto train(OnEpoch&& callback) -> Result {
        return *std::ranges::find_if(*this, [&](auto&& result) { 
            callback(result);
            return result.isTrained(); 
        });
    }

    auto trainee() -> MLPerceptron& { return trainee_; }

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

        std::ranges::copy(std::views::iota(0u, indexes_.size()), indexes_.begin());

        auto rnd = Random();

        rnd.separated(0.3f, 0.2f, trainee.weights());
        rnd.separated(0.3f, 0.2f, trainee.biases());
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

    mutable std::vector<float> errors_;
    mutable std::vector<float> derived_;
    mutable std::vector<int> indexes_;

    MLPerceptron trainee_;
    float learnrate_;
    float error_;
    int maxEpochs_;
    Dataset trainset_;
    Dataset testset_;
    derivation_function derivation_;
};

}