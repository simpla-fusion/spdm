#include "db/Entry.h"
#include "db/XPath.h"
#include "utility/Logger.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("Create Node", "[SpDB]")
{
    using namespace sp::db::literals;

    sp::db::Entry entry;

    entry["A"s].as<std::string>("1234");
    entry["B"s].as<std::string>("5678");

    REQUIRE(entry.type() == sp::db::entry_value_type_tags::Object);

    REQUIRE(entry.as_object().size() == 2);

    entry["C"].resize(4);

    REQUIRE(entry["C"].as_array().size() == 4);

    REQUIRE(entry["C"].type() == sp::db::entry_value_type_tags::Array);
    REQUIRE(entry["C"].size() == 4);

    entry["C"][2] = 12344.56;
    entry["C"][3] = 6.0 + 4.0i;

    entry["C"].push_back().as<int>(135);
    entry["C"].push_back().as<float>(6.0);
    entry["C"].push_back().as<std::string>("3.1415926");

    REQUIRE(entry["C"].size() == 7);

    REQUIRE(entry["C"][2].as<double>() == 12344.56);
    REQUIRE(entry["C"][2].as<int>() == 12344);
    REQUIRE(entry["C"][4].as<std::string>() == "135");
    REQUIRE(entry["C"][6].as<double>() == 3.1415926);

    std::string message = "hello world!";

    entry["D/E/F"_p] = message;

    REQUIRE(entry["D"]["E"]["F"].as<std::string>() == message);

    std::cout << entry << std::endl;

    // std::cout << "====================================" << std::endl;
    // entry.as_table()["C"].push_back().as_scalar().set_value<std::string>("1234");

    // // entry.set_value<std::string>("1234");
    // std::cout << entry << std::endl;

    // // std::cout << "====================================" << std::endl;

    // // entry.append().set_value<std::string>("4567");
    // std::cout << "====================================" << std::endl;

    // entry.as_array().push_back().as_scalar().set_value<std::string>("7890");

    // std::cout << entry << std::endl;

    // REQUIRE(entry.child("C").child(0).get_value<std::string>() == "1234");
}