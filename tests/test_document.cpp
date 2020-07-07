#include "SpNode.h"

#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("SpDocument Create", "[SpDB]")
{

    sp::Node node;

    node.attributes().set("key", "a");
}