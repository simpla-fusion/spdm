#include "Node.h"
#include "utility/Logger.h"
#include <iostream>
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
const char PLUGIN_NAME[] = "xml";

// TEST_CASE("Attribute ", "[SpDB]")
// {
//     sp::Node node(PLUGIN_NAME);

//     node.set_attribute<std::string>("A", ("a"));

//     node.set_attribute<std::string>("B", std::string("b"));

//     REQUIRE(node.has_attribute("A") == true);

//     REQUIRE(node.has_attribute("C") == false);

//     REQUIRE(node.get_attribute<std::string>("A") == "a");
// }
TEST_CASE("Object", "[SpDB]")
{
    sp::Node node(PLUGIN_NAME);

    node["A"].set_value<std::string>("1234");

    node["B"].set_value<sp::Entry::ElementType::Float>(3.14);

    // node["D/E/F"].set_value<double>(1.2345);

    // REQUIRE(node.size() == 3);
    REQUIRE(node["A"].get_value<sp::Entry::ElementType::String>() == "1234");
    
    REQUIRE(node["B"].get_value<sp::Entry::ElementType::Float>() == 3.14);

    // REQUIRE(node["D"]["E"]["F"].get_value<sp::Entry::ElementType::Float>() == 1.2345);
}
// TEST_CASE("Array", "[SpDB]")
// {
//     sp::Node node(PLUGIN_NAME);

//     REQUIRE(node.size() == 2);

//     node["C"][-1].set_value<sp::Entry::ElementType::Integer>(5);

//     node["C"][-1].set_value<double>(6.0);

//     REQUIRE(node["C"].size() == 2);

//     REQUIRE(node["C"][0].get_value<sp::Entry::ElementType::Integer>() == 5);
//     REQUIRE(node["C"][1].get_value<sp::Entry::ElementType::Float>() == 6.0);
// }