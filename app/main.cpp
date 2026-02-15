#include <YetAnotherMlp/MLPTrainer.hpp>
#include <YetAnotherMlp/Mnist.hpp>

#include <SFML/Graphics.hpp>

#include <memory>
#include <thread>

namespace {

auto window() -> std::unique_ptr<sf::RenderWindow> {
    const auto title = "YetAnotherMlp";
    const auto videoMode = sf::VideoMode(1000, 100);
    const auto framerate = 60;

    auto window = std::make_unique<sf::RenderWindow>(videoMode, title);
    window->setFramerateLimit(framerate);
    return window;
}

}

struct HeatMap: sf::Drawable, sf::Transformable {
    HeatMap(
        sf::Vector2u resolution, 
        sf::Vector2f size,
        std::span<const float> values
    ) {
        rects_.resize(resolution.x * resolution.y * 4);
        rects_.setPrimitiveType(sf::Quads);
        resolution_ = resolution;
        size_ = size;

        update(values);
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override {
        states.transform *= getTransform();
        target.draw(rects_, states);
    }

    template<std::ranges::forward_range Values>
    requires(std::floating_point<std::ranges::range_value_t<Values>>)
    void update(Values&& values) {
        using namespace std::views;

        const auto min = std::ranges::min(values);
        const auto max = std::ranges::max(values);
        const auto range = std::max(std::fabs(max - min), decltype(max)(1));
        const auto rectSize = sf::Vector2f { 
            size_.x / resolution_.x, 
            size_.y / resolution_.y 
        };

        for(const auto [i, value] : zip(iota(0), values)) {
            const auto row = i / resolution_.x;
            const auto col = i % resolution_.x;
            const auto brightness = 255.0f * (value - min) / range;
            const auto color = sf::Color(brightness, brightness, brightness);

            rects_[i * 4 + 0].position = {rectSize.x * (col + 0), rectSize.y * (row + 0)};
            rects_[i * 4 + 1].position = {rectSize.x * (col + 1), rectSize.y * (row + 0)};
            rects_[i * 4 + 2].position = {rectSize.x * (col + 1), rectSize.y * (row + 1)};
            rects_[i * 4 + 3].position = {rectSize.x * (col + 0), rectSize.y * (row + 1)};

            rects_[i * 4 + 0].color = color;
            rects_[i * 4 + 1].color = color;
            rects_[i * 4 + 2].color = color;
            rects_[i * 4 + 3].color = color;
        }
    }

private:
    sf::VertexArray rects_;
    sf::Vector2u resolution_;
    sf::Vector2f size_;
};

auto mnistTrainer() -> yam::MLPTrainer {
        const auto trainset = yam::Mnist::read(
        "../resources/train-images.idx3-ubyte", 
        "../resources/train-labels.idx1-ubyte"
    );

    const auto testset = yam::Mnist::read(
        "../resources/t10k-images.idx3-ubyte", 
        "../resources/t10k-labels.idx1-ubyte"
    );
    
    auto mlp = yam::MLPerceptron(
        {trainset.inputSize(), trainset.outputSize()}, 
        false, 
        yam::Activation::sigmoid
    );

    return yam::MLPTrainer(mlp, 0.1, 0.035, 20, trainset, testset, yam::Derivation::sigmoid);
}

auto heatMaps(const yam::MLPerceptron& mlp) -> std::vector<HeatMap> {
    const auto layerSize = mlp.topology()[mlp.topology().size() - 2];
    const auto resolution = sf::Vector2u(
        std::sqrt(layerSize), 
        std::sqrt(layerSize)
    );

    auto heatMaps = std::vector<HeatMap>();
    for (auto d = 0; d < 10; ++d) {
        heatMaps.emplace_back(HeatMap(
            resolution,
            {90, 90},
            mlp.weights().subspan(d * layerSize, layerSize)
        ));
        heatMaps.back().setPosition(d * 100 + 5, 5);
    }
    return heatMaps;
}

int main() {
    // std::cout << "iteration: " << result.epoch << ", error: " << result.error << "\n";
    auto trainer = mnistTrainer();
    auto first = trainer.begin();
    auto mlp = *first->trainee;
    auto heatMaps = ::heatMaps(mlp);

    const auto interval = std::chrono::seconds(2);

    auto time = std::chrono::high_resolution_clock::now();

    for (auto window = ::window(); window->isOpen(); window->display()) {
        const auto now = std::chrono::high_resolution_clock::now();
        if ((now - time > interval) && !first->isTrained()) {
            time = now;
            ++first;
            const auto result = *first;
            std::cout << "iteration: " << result.epoch << ", error: " << result.error << "\n";
            heatMaps = ::heatMaps(*first->trainee);
        }

        for (auto event = sf::Event(); window->pollEvent(event);) {
            if (event.type == sf::Event::Closed) {
                window->close();
            }
        }

        window->clear();
        for (const auto map : heatMaps) {
            window->draw(map);
        }
    }
    return 0;
}