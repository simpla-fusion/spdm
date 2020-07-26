#include "HierarchicalTree.h"
#include "utility/Logger.h"
#include <iostream>
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("HData ", "[HierarchicalTree]")
{
    sp::HierarchicalTree<> attr;

    attr["A"] = 1234.5;

    REQUIRE(attr["A"].template get_value<double>() == 1234.5);
}

TEST_CASE("Object", "[HierarchicalTree]")
{
    sp::HierarchicalTree<> node;
    using namespace std::literals;

    node["A"] = "1234"s;

    node["B"].template set_value<sp::DataType::Float>(3.14);

    REQUIRE(node.size() == 2);

    REQUIRE(node["A"].template get_value<sp::DataType::String>() == "1234");

    REQUIRE(node["B"].template get_value<sp::DataType::Float>() == 3.14f);
}
TEST_CASE("Path", "[HierarchicalTree]")
{
    sp::HierarchicalTree<> node;

    node["D/E/F"].template set_value<double>(1.2345);

    REQUIRE(node["D"]["E"]["F"].template get_value<sp::DataType::Double>() == 1.2345);
}
// TEST_CASE("Array", "[HierarchicalTree]")
// {
//     sp::HierarchicalTree node;

//     node["C"].resize(2);

//     node["C"][1] = (5);

//     node["C"][0].emplace<double>(6.0);

//     REQUIRE(node["C"].size() == 2);

//     REQUIRE(node["C"][1].get_value<sp::DataType::Integer>() == 5);
//     REQUIRE(node["C"][0].get_value<sp::DataType::Double>() == 6.0);
// }