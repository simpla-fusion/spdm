#include "db/Document.h"
#include "db/Entry.h"
#include "utility/Factory.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("SpDocument Create", "[SpDB]")
{
    using namespace sp::db;
    using namespace sp::db::literals;

    auto mapper = sp::db::NodeObject::create("file://./mapper/EAST/imas/3");
    sp::db::Entry entry{{"mapper", mapper},
                        {"base", "mdsplus://./tests/data"}};

    // std::cout << entry << std::endl;

    std::cout << entry["ids/timeslice"][0]["ne"].get_value<double>() << std::endl;
    std::cout << entry["ids/timeslice[@id=1]/ne"].get_value<double>() << std::endl;

    // std::cout << Factory<EntryInterface, Entry*, const std::string&, Entry*>::counter << std::endl;
    // entry.set_attribute("A", std::string("a"));
    // entry.set_attribute("B", std::string("b"));
    // entry["A"].as<std::string>("1234");
    // entry["B"].as<std::string>("5678");

    // entry["C"][-1].set_value<int>(5);
    // entry["C"][-1].set_value<float>(6.0);

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