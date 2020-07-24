#include "Node.h"
#include "utility/Logger.h"
#include <iostream>
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("Create Node", "[SpDB]")
{
    sp::logger::set_stdout_level(-1000);

    sp::Node node;

    node.set_attribute("A", std::string("a"));
    node.set_attribute("B", std::string("b"));
    node["A"].set_value<std::string>("1234");
    node["B"].set_value<std::string>("5678");

    node["C"][-1].set_value<int>(5);
    node["C"][-1].set_value<float>(6.0);

    std::cout << node.size() << std::endl;

    for (auto it = node.first_child(); !it.is_null(); ++it)
    {
        std::cout << it->name() << ":" << *it << std::endl;
    }

    // std::cout << "====================================" << std::endl;
    // entry.as_table()["C"].as_array().push_back().as_scalar().set_value<std::string>("1234");

    // // entry.set_value<std::string>("1234");
    // std::cout << entry << std::endl;

    // // std::cout << "====================================" << std::endl;

    // // entry.append().set_value<std::string>("4567");
    // std::cout << "====================================" << std::endl;

    // entry.as_array().push_back().as_scalar().set_value<std::string>("7890");

    // std::cout << entry << std::endl;

    // REQUIRE(entry.child("C").child(0).get_value<std::string>() == "1234");
}