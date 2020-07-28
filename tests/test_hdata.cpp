#include "HierarchicalTree.h"
#include "HierarchicalTreeAlgorithm.h"
#include "utility/Logger.h"
#include <iostream>
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
using namespace std::literals;
TEST_CASE("HData ", "[HierarchicalTree]")
{
    sp::HierarchicalNode attr;

    attr["A"] = 1234.5;

    REQUIRE(attr["A"].template get_value<double>() == 1234.5);
}

TEST_CASE("Object", "[HierarchicalTree]")
{
    sp::HierarchicalNode node;

    node["A"] = "1234"s;

    node["B"].template set_value<sp::HierarchicalNode::DataType::Float>(3.14);

    node["C"] = std::array<int, 3>{1, 2, 3};

    std::cout << node << std::endl;

    REQUIRE(node.size() == 3);

    REQUIRE(node["A"].template get_value<sp::HierarchicalNode::DataType::String>() == "1234");

    REQUIRE(node["B"].template get_value<sp::HierarchicalNode::DataType::Float>() == 3.14f);
}
// TEST_CASE("Path", "[HierarchicalTree]")
// {
//     sp::HierarchicalNode node;

//     node["D/E/F"].template set_value<double>(1.2345);

//     REQUIRE(node["D"]["E"]["F"].template get_value<sp::HierarchicalNode::DataType::Double>() == 1.2345);
// }
TEST_CASE("Array", "[HierarchicalTree]")
{
    sp::HierarchicalNode node;

    node["C"].resize(2);

    node["C"][1] = (5);

    node["C"][0].set_value<double>(6.0);
    
    std::cout << node << std::endl;

    REQUIRE(node["C"].size() == 2);

    REQUIRE(node["C"][1].get_value<sp::HierarchicalNode::DataType::Int>() == 5);
    REQUIRE(node["C"][0].get_value<sp::HierarchicalNode::DataType::Double>() == 6.0);
}