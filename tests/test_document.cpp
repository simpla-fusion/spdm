#include "Node.h"

#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("SpDocument Create", "[SpDB]")
{
    // struct A
    // {
    //     virtual void foo() { std::cout << "I'm A" << std::endl; }
    // };
    // struct B : public A
    // {
    //     void foo()  { std::cout << "I'm B" << std::endl; }
    // };

    // A *p = new B;
    // p->foo();
    sp::Node node;

    node.attribute("A", "a");
    node.attribute("B", "1234");
    node.child("C").as_scalar(std::any(std::string("1234")));

    REQUIRE(std::any_cast<std::string>(node.child("C").as_scalar()) == "1234");
}