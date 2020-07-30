#include "db/Node.h"
#include "utility/Logger.h"
#include <iostream>
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
const char PLUGIN_NAME[] = "memory";
using namespace std::literals;
namespace spdb = sp::db;

TEST_CASE("Attribute ", "[SpDB]")
{
    spdb::Node node(PLUGIN_NAME);

    node["A"].set_attribute<std::string>("A", "a");

    node["A"].set_attribute<double>("B", 12.345);

    REQUIRE(node["A"].has_attribute("A") == true);

    REQUIRE(node["A"].has_attribute("C") == false);

    REQUIRE(node["A"].get_attribute<std::string>("A") == "a");
}
TEST_CASE("Object", "[SpDB]")
{
    spdb::Node node(PLUGIN_NAME);

    node["A"].set_value<std::string>("1234");

    node["B"].set_value<spdb::Node::type_tags::Float>(3.14);

    node["D/E/F"].set_value<double>(1.2345);

    REQUIRE(node.size() == 3);

    REQUIRE(node["A"].get_value<spdb::Node::type_tags::String>() == "1234");

    REQUIRE(node["B"].get_value<spdb::Node::type_tags::Float>() == 3.14f);

    REQUIRE(node["D"]["E"]["F"].get_value<spdb::Node::type_tags::Double>() == 1.2345);
}
TEST_CASE("Array", "[SpDB]")
{
    spdb::Node node(PLUGIN_NAME);

    node["C"][-1].set_value<spdb::Node::type_tags::Int>(5);

    node["C"][-1].set_value<double>(6.0);

    REQUIRE(node["C"].size() == 2);

    REQUIRE(node["C"][0].get_value<spdb::Node::type_tags::Int>() == 5);
    REQUIRE(node["C"][1].get_value<spdb::Node::type_tags::Float>() == 6.0);
}