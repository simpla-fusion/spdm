#include "db/Entry.h"
#include "db/XPath.h"
#include "utility/Logger.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

using namespace sp::db::literals;

TEST_CASE("Object", "[SpDB:Entry]")
{
    sp::db::Node opt{{"B"s, {{"b", 1}, {"c", "hello world"}}}};

    sp::db::Entry entry(opt);

    VERBOSE << entry;

    REQUIRE(entry["B"]["b"].get_value<int>() == 1);

    REQUIRE(entry["B"]["c"].get_value<std::string>() == "hello world");

    entry["A"s].set_value<std::string>("1234");
    entry["B"s].set_value<std::string>("5678");

    REQUIRE(entry.type() == sp::db::Node::tags::Object);

    REQUIRE(entry.count() == 2);
}

TEST_CASE("Path", "[SpDB:Array]")
{
    sp::db::Entry entry;

    std::string message = "hello world!";

    entry["D/E/F"_p] = message;

    REQUIRE(entry["D"]["E"]["F"].get_value<std::string>() == message);

    VERBOSE << entry;
}

TEST_CASE("Array", "[SpDB:Entry]")
{
    sp::db::Entry entry;

    entry["C"].resize(4);

    REQUIRE(entry["C"].count() == 4);
    REQUIRE(entry["C"].type() == sp::db::Node::tags::Array);

    entry["C"][2] = 12344.56;
    entry["C"][3] = 6.0 + 4.0i;

    entry["C"].push_back().set_value<int>(135);
    entry["C"].push_back().set_value<float>(6.0);
    entry["C"].push_back().set_value<std::string>("3.1415926");

    REQUIRE(entry["C"].count() == 7);

    REQUIRE(entry["C"][2].get_value<double>() == 12344.56);
    REQUIRE(entry["C"][4].get_value<std::string>() == "135");
    REQUIRE(entry["C"][6].get_value<double>() == 3.1415926);

    VERBOSE << entry;
}
