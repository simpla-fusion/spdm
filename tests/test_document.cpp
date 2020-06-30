// #include "SpDocument.h"
#include "SpRange.h"
#include "SpDocument.h"

#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("SpDocument Create", "[SpDB]")
{

    // SpNode node;

    // node.attribute("first") = 1234;

    int d[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto range = make_range(d + 0, d + 10);

    for (auto const &v : range)
    {
        std::cout << "[RANGE]" << v << std::endl;
    }

    auto range2 = range.filter([](int const &d) { return d % 2 == 0; });

    for (auto const &v : range2)
    {
        std::cout << "[RANGE2]" << v << std::endl;
    }

    SpNode root;

    for (auto const &a : root.attributes())
    {
        std::cout << dynamic_cast<SpAttribute const &>(a).name() << "=" << std::any_cast<std::string>(dynamic_cast<SpAttribute const &>(a).value()) << std::endl;
    }
}