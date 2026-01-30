#include <YetAnotherMlp/MLPTrainer.hpp>
#include <YetAnotherMlp/Mnist.hpp>

#include <SFML/Graphics.hpp>

#include <memory>

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

struct HeatMap: sf::Drawable {
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
        target.draw(rects_, states);
    }

    void update(std::span<const float> values) {
        const auto min = std::ranges::min(values);
        const auto max = std::ranges::max(values);
        const auto rectSize = sf::Vector2f { 
            size_.x / resolution_.x, 
            size_.y / resolution_.y 
        };
        for (auto i = 0u; i < values.size(); ++i) {
            const auto row = i / resolution_.x;
            const auto col = i % resolution_.y;
            const auto brightness = 255.0f * (values[i] - min) / (max - min);
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

auto mnist() -> yam::MLPerceptron {
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
        true, 
        yam::Activation::sigmoid
    );

    const auto trainer = yam::MLPTrainer();

    trainer.train(mlp, 0.1, 0.011, 1000, trainset, testset, yam::Derivation::sigmoid);

    return mlp;
}

int main() {
    const auto mlp = mnist();

    const auto resolution = sf::Vector2u(280, 28);
    auto values = std::vector<float>(resolution.x * resolution.y);

    auto weightsToValues = [](
        std::span<const float> weights,
        std::span<      float> values
    ) {
        auto weight = weights.begin();
        for (auto o = 0; o < 28 * 28; ++o) {
            for (auto i = 0; i < 10; ++i) {
                values[i * 28 * 28 + o] = weight[i * 28 * 28];
            }
            weight++;
        }
    };

    weightsToValues(mlp.weights(), values);

    const auto heatMap = HeatMap(
        resolution,
        {1000, 100},
        mlp.weights()
    );

    for (auto window = ::window(); window->isOpen(); window->display()) {
        for (auto event = sf::Event(); window->pollEvent(event);) {
            if (event.type == sf::Event::Closed) {
                window->close();
            }
        }

        window->clear();
        window->draw(heatMap);
    }
    return 0;
}