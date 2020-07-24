#include "Node.h"
#include "utility/Logger.h"
#include <iostream>
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("Create Node", "[SpDB]")
{
    sp::logger::set_stdout_level(-1000);

    sp::Node node;

    node.set_attribute<std::string>("A", ("a"));
    node.set_attribute<std::string>("B", std::string("b"));
    node["A"].set_value<std::string>("1234");
    node["B"].set_value<std::string>("5678");

    node["C"][-1].set_value<int>(5);
    node["C"][-1].set_value<double>(6.0);
    node["D/E/F"].set_value<double>(1.2345);

    std::cout << node << std::endl;

    REQUIRE(node.get_attribute<std::string>("A") == "a");

    REQUIRE(node.size() == 6);

    REQUIRE(node["C"].size() == 2);

    REQUIRE(node["D"]["E"]["F"].get_value<double>() == 1.2345);

    // for (auto it = node.first_child(); !it.is_null(); ++it)
    // {
    //     std::cout << it->name() << ":" << *it << std::endl;
    // }

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