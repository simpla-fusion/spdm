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

    node.as<double>(3.1415926);
    REQUIRE(node.as<int>() == 3);

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

    sp::db::Node node({{"B"s, {{"b", 1}, {"c", "hello world"}}}});

    VERBOSE << node;

    auto& obj = node.as_object();

    REQUIRE(obj.fetch(sp::db::Path::parse("B/b")).as<int>() == 1);

    REQUIRE(obj.fetch(sp::db::Path::parse("B/c")).as<std::string>() == "hello world");

    REQUIRE(obj.fetch(sp::db::Path::parse("B"), {{"$type", "1"}}).as<int>() == sp::db::Node::tags::Object);

    REQUIRE(obj.fetch(sp::db::Path::parse("B"), {{"$count", "1"}}).as<int>() == 2);

    obj.update(sp::db::Path::parse("C/A"), "Hello world!");

    REQUIRE(obj.fetch(sp::db::Path::parse("C/A")).as<std::string>() == "Hello world!");

    REQUIRE(obj.update(sp::db::Path::parse("C/B"), 3.1415926).as<double>() == 3.1415926);

    REQUIRE(obj.fetch(sp::db::Path::parse("C/B")).as<double>() == 3.1415926);

    REQUIRE(obj.update(sp::db::Path::parse("C/B"), "Hello world!").as<std::string>() == "Hello world!");

    REQUIRE(obj.update(sp::db::Path::parse("C/B"), {{"$default", 5678}}).as<std::string>() == "Hello world!");

    REQUIRE(obj.update(sp::db::Path::parse("C/B"), 5678).as<int>() == 5678);

    VERBOSE << obj;
}
