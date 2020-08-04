#include "db/Cursor.h"

#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("Cursor:", "[SpDB]")
{
    std::vector<int> v = {1, 2, 3, 4, 5};

    auto c = sp::db::make_cursor(v.begin(), v.end());

    do
    {
        std::cout << *c << " ";
    } while (c.next());

    std::cout << std::endl;
}

TEST_CASE("Cursor:Map", "[SpDB]")
{
    std::map<std::string, int> v = {{"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}};

    auto c = sp::db::make_cursor(v.begin(), v.end());

    auto d = c.map<int>();

    do
    {
        std::cout << c->first << ":" << c->second << " ";

    } while (c.next());

    std::cout << std::endl;

    do
    {
        std::cout << *d << " ";

    } while (d.next());

    std::cout << std::endl;
}