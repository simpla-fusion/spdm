#include "db/Entry.h"
#include "utility/Logger.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("Create Node", "[SpDB]")
{
    sp::logger::set_stdout_level(-1000);

    using namespace sp::db;
    sp::db::Entry entry;

    // entry.set_attribute("A", std::string("a"));
    // entry.set_attribute("B", std::string("b"));
    // entry["A"].set_value<std::string>("1234");
    entry["B"].as<std::string>("5678");
    // entry["C"].as_array().resize(4);
    entry["C"].push_back().as<int>(5);
    entry["C"].push_back().as<float>(6.0);

    std::cout << entry << std::endl;

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