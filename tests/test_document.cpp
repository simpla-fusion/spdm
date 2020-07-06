// #include "SpDocument.h"
#include "SpDB.h"
#include "SpUtil.h"

#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("SpDocument Create", "[SpDB]")
{

    // SpNode node;

    // node.attribute("first") = 1234;

    // int d[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // auto range = make_range(d + 0, d + 10);

    // for (auto const &v : range)
    // {
    //     std::cout << "[RANGE]" << v << std::endl;
    // }

    // auto range2 = range.filter([](int const &d) { return d % 2 == 0; });

    // for (auto const &v : range2)
    // {
    //     std::cout << "[RANGE2]" << v << std::endl;
    // }

    sp::SpDocument doc;
    doc.load("/workspaces/SpDB/mapper/EAST/imas/3/config.xml");

    auto node = doc.root().child("first");

    for (auto const &a : doc.root().attributes())
    {
        std::cout << a << std::endl;
    }
    for (auto const &n : doc.root().children())
    {
        std::cout << n << std::endl;
    }
}