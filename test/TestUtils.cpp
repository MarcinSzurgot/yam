#include <YetAnotherMlp/Utils.hpp>

#include <gtest/gtest.h>

TEST(TestUtils, EmptyAdjacentElements) {
    const auto topology = std::vector<int>();

    const auto adjacent = yam::adjacent(topology);

    ASSERT_EQ(adjacent.size(), 0);
}

TEST(TestUtils, AdjacentElements) {
    const auto topology = std::vector { 1, 2, 3, 4 };

    const auto adjacent = yam::adjacent(topology);

    ASSERT_EQ(adjacent.size(), 3);

    ASSERT_EQ(adjacent[0].first, 1);
    ASSERT_EQ(adjacent[0].second, 2);

    ASSERT_EQ(adjacent[1].first, 2);
    ASSERT_EQ(adjacent[1].second, 3);

    ASSERT_EQ(adjacent[2].first, 3);
    ASSERT_EQ(adjacent[2].second, 4);
}

TEST(TestUtils, Sum) {
    const auto numbers = std::vector {3, 5, 7};
    const auto sum = yam::sum(numbers);
    ASSERT_EQ(sum, 15);
}

TEST(TestUtils, EmptySum) {
    const auto numbers = std::vector<float>();
    const auto sum = yam::sum(numbers);
    ASSERT_EQ(sum, 0.0f);
}