#include "db/Cursor.h"

#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("Cursor:", "[SpDB]")
{
    std::vector<int> v = {1, 2, 3, 4, 5};

    auto c = sp::db::Cursor<int>(v.begin(), v.end());

    do
    {
        std::cout << *c << " ";
    } while (c.next());

    std::cout << std::endl;
}

TEST_CASE("Cursor:Map", "[SpDB]")
{
    std::map<std::string, int> v = {{"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}};
    // std::vector<int> v = {1, 2, 3, 4, 5};
    auto c = sp::db::Cursor<const int>(v.cbegin(), v.cend(), [](auto&& item) -> const int& { return item.second; });

    // auto d = c.map<int>();

    // do
    // {
    //     std::cout << c->first << ":" << c->second << " ";

    // } while (c.next());

    // std::cout << typeid(std::iterator_traits<decltype(v.begin())>::pointer).name() << std::endl;
    // std::cout << std::boolalpha << std::is_same_v<std::iterator_traits<decltype(v.cbegin())>::value_type, std::pair<const std::string, int>> << std::endl;

    // std::cout << std::boolalpha << std::is_same_v<std::iterator_traits<decltype(v.cbegin())>::pointer, typename sp::db::CursorProxy<const std::iterator_traits<decltype(v.cbegin())>::value_type>::pointer> << std::endl;

    do
    {
        std::cout << *c << " ";

    } while (c.next());

    std::cout << std::endl;
}