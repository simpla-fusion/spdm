#include "HierarchicalData.h"
#include "utility/Logger.h"
#include <iostream>
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("HData ", "[HierarchicalData]")
{
    sp::HierarchicalData attr;

    attr["A"] = 1234.5;

    REQUIRE(attr["A"].as<double>() == 1234.5);
}

TEST_CASE("Object", "[HierarchicalData]")
{
    sp::HierarchicalData node;

    node["A"].emplace<std::string>("1234");

    node["B"].emplace<sp::DataType::Float>(3.14);

    REQUIRE(node.size() == 2);

    REQUIRE(node["A"].as<sp::DataType::String>() == "1234");

    REQUIRE(node["B"].as<sp::DataType::Float>() == 3.14f);
}
TEST_CASE("Path", "[HierarchicalData]")
{
    sp::HierarchicalData node;

    node["D/E/F"].emplace<double>(1.2345);

    REQUIRE(node["D"]["E"]["F"].as<sp::DataType::Double>() == 1.2345);
}
// TEST_CASE("Array", "[HierarchicalData]")
// {
//     sp::HierarchicalData node;

//     node["C"].resize(2);

//     node["C"][1] = (5);

//     node["C"][0].emplace<double>(6.0);

//     REQUIRE(node["C"].size() == 2);

//     REQUIRE(node["C"][1].as<sp::DataType::Integer>() == 5);
//     REQUIRE(node["C"][0].as<sp::DataType::Double>() == 6.0);
// }