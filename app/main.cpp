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

    void update(std::span<const float> values) {
        const auto min = std::ranges::min(values);
        const auto max = std::ranges::max(values);
        const auto rectSize = sf::Vector2f { 
            size_.x / resolution_.x, 
            size_.y / resolution_.y 
        };
        for (auto i = 0u; i < values.size(); ++i) {
            const auto row = i / resolution_.x;
            const auto col = i % resolution_.x;
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
        false, 
        yam::Activation::sigmoid
    );

    auto trainer = yam::MLPTrainer(
        mlp, std::move(testset), std::move(trainset), yam::Derivation::sigmoid, 0.1, 0.035, 5
    );

    trainer.train();

    return std::move(trainer.trainee());
}

int main() {
    const auto mlp = mnist();
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

    for (auto window = ::window(); window->isOpen(); window->display()) {
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