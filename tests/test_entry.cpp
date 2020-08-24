#include "db/Entry.h"
#include "db/XPath.h"
#include "utility/Logger.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("Create Node", "[SpDB]")
{
    using namespace sp::db::literals;
    // using namespace std::complex_literals;
    // using namespace std::string_literals;

    sp::db::Entry entry;

    entry["A"s].as<std::string>("1234");
    entry["B"s].as<std::string>("5678");

    entry["C"].resize(4);
    entry["C"][2] = 12344.56;
    entry["C"][3] = 6.0 + 4.0i;
    entry["C"].push_back().as<int>(5);
    entry["C"].push_back().as<float>(6.0);

    // entry["D/E/F"_p] = "hello world!";

    VERBOSE << sizeof(sp::db::Entry);

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