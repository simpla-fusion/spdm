#include "Node.h"

#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("SpDocument Create", "[SpDB]")
{
    sp::Node node;

    // node.set_attribute("A", "a");
    // node.set_attribute("B", "1234");
    node.child("A").set_value<std::string>("1234");
    node.child("B").set_value<std::string>("1234");
    node.child("C").set_value<std::string>("1234");

    // node.append().set_value<int>(5);
    std::cout << node << std::endl;
    // REQUIRE(node.child("C").child(0).get_value<std::string>() == "1234");
}