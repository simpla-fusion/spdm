#include "Entry.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("SpDocument Create", "[SpDB]")
{

    auto p = sp::create_entry();

    // p->set_attribute("A", "a");
    // // node.set_attribute("B", "1234");
    // node["A"].set_value<std::string>("1234");
    // node["B"].set_value<std::string>("1234");
    p->as<sp::TypeTag::Array>()
        ->push_back()
        ->as<sp::TypeTag::Scalar>()
        ->set_string("1234");
    // std::cout << node << std::endl;
    // std::cout << "====================================" << std::endl;
    // node.as_table()["C"].as_array().push_back().as_scalar().set_value<std::string>("1234");
}