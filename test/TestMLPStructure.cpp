#include <YetAnotherMlp/MLPStructure.hpp>

#include <gtest/gtest.h>

// TEST(TestMLPStructure, NeuronsCountsInLayers) {
//     auto structure = yam::MLPStructure({3, 4, 5, 6}, false);

//     ASSERT_EQ(structure.neurons().rows().size(), 4);
//     ASSERT_EQ(structure.neurons().row(0).size(), 3);
//     ASSERT_EQ(structure.neurons().row(1).size(), 4);
//     ASSERT_EQ(structure.neurons().row(2).size(), 5);
//     ASSERT_EQ(structure.neurons().row(3).size(), 6);
// }

// TEST(TestMLPStructure, BiasesCountsInLayers) {
//     auto structure = yam::MLPStructure({3, 4, 5, 6}, true);

//     ASSERT_EQ(structure.biases().rows().size(), 3);
//     ASSERT_EQ(structure.biases().row(0).size(), 4);
//     ASSERT_EQ(structure.biases().row(1).size(), 5);
//     ASSERT_EQ(structure.biases().row(2).size(), 6);
// }

// TEST(TestMLPStructure, WeightsCountsInLayers) {
//     auto structure = yam::MLPStructure({3, 4, 5, 6}, true);

//     ASSERT_EQ(structure.weights().rows().size(), 3);
//     ASSERT_EQ(structure.weights().row(0).size(), 3 * 4);
//     ASSERT_EQ(structure.weights().row(1).size(), 4 * 5);
//     ASSERT_EQ(structure.weights().row(2).size(), 5 * 6);
// }

// TEST(TestMLPStructure, EmptyStructure) {
//     auto structure = yam::MLPStructure(std::vector<int>(), false);

//     ASSERT_TRUE(structure.neurons().values().empty());
//     ASSERT_TRUE(structure.weights().values().empty());
//     ASSERT_TRUE(structure.biases().values().empty());
// }

// TEST(TestMLPStructure, EmptyBiases) {
//     auto structure = yam::MLPStructure({3, 3, 3}, false);

//     ASSERT_FALSE(structure.neurons().values().empty());
//     ASSERT_FALSE(structure.weights().values().empty());
//     ASSERT_TRUE(structure.biases().values().empty());
// }