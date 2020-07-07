#include "SpDocument.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("SpDocument Create", "[SpDB]")
{

    sp::SpNode node;

    node.attribute("key", "a");
}