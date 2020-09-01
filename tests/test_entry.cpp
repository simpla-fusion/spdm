#include "db/Entry.h"
#include "db/XPath.h"
#include "utility/Logger.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

using namespace sp::db::literals;
TEST_CASE("Node ", "[SpDB]")
{
    sp::db::Node node;
    REQUIRE(node.type() == sp::db::Node::tags::Null);

    node.as<int>(124);
    REQUIRE(node.as<int>() == 124);

    node.as<sp::db::Node::tags::String>("3.1415926");
    REQUIRE(node.as<double>() == 3.1415926);

    node.clear();
    REQUIRE(node.type() == sp::db::Node::tags::Null);

    node.as_object();

    REQUIRE(node.type() == sp::db::Node::tags::Object);

    node.clear();
    node.as_array();
    REQUIRE(node.type() == sp::db::Node::tags::Array);
}

TEST_CASE("NodeObject", "[SpDB]")
{

    sp::db::NodeObject obj({{"B"s, {{"b", 1}, {"c", "hello world"}}}});

    REQUIRE(obj.fetch(sp::db::Path::parse("B/b"), {}).as<int>() == 1);

    REQUIRE(obj.fetch(sp::db::Path::parse("B/c"), {}).as<std::string>() == "hello world");

    obj.update(sp::db::Path::parse("C/A"), "Hello world!");

    REQUIRE(obj.fetch(sp::db::Path::parse("C/A"), {}).as<std::string>() == "Hello world!");

    REQUIRE(obj.merge(sp::db::Path::parse("C/A"), {}).as<std::string>() == "Hello world!");

    REQUIRE(obj.merge(sp::db::Path::parse("C/B"), 3.1415926).as<double>() == 3.1415926);

    REQUIRE(obj.fetch(sp::db::Path::parse("C/B"), {}).as<double>() == 3.1415926);

    REQUIRE(obj.fetch(sp::db::Path::parse("C/C"), 123).as<int>() == 123);

    std::cout << obj << std::endl;
}

// TEST_CASE("Entry ", "[SpDB]")
// {
//     sp::db::Entry entry({{"B"s, {{"b", 1}, {"c", "hello world"}}}});

//     REQUIRE(entry["B"]["b"].as<int>() == 1);

//     REQUIRE(entry["B"]["c"].as<std::string>() == "hello world");

//     // entry["A"s].as<std::string>("1234");
//     // entry["B"s].as<std::string>("5678");

//     // REQUIRE(entry.type() == sp::db::Node::tags::Object);

//     // REQUIRE(entry.as_object().size() == 2);

//     // entry["C"].resize(4);

//     // REQUIRE(entry["C"].as_array().size() == 4);
//     // REQUIRE(entry["C"].type() == sp::db::Node::tags::Array);
//     // REQUIRE(entry["C"].size() == 4);

//     // entry["C"][2] = 12344.56;
//     // entry["C"][3] = 6.0 + 4.0i;
//     // std::cout << entry << std::endl;

//     // entry["C"].push_back().as<int>(135);
//     // entry["C"].push_back().as<float>(6.0);
//     // entry["C"].push_back().as<std::string>("3.1415926");
//     // std::cout << entry << std::endl;

//     // // REQUIRE(entry["C"].size() == 7);

//     // REQUIRE(entry["C"][2].as<double>() == 12344.56);
//     // REQUIRE(entry["C"][2].as<int>() == 12344);
//     // REQUIRE(entry["C"][4].as<std::string>() == "135");
//     // REQUIRE(entry["C"][6].as<double>() == 3.1415926);

//     // std::string message = "hello world!";

//     // entry["D/E/F"_p] = message;

//     // REQUIRE(entry["D"]["E"]["F"].as<std::string>() == message);

//     // std::cout << entry << std::endl;
// }