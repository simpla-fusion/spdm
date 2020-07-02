//
// Created by salmon on 18-1-22.
//
#include <boost/any.hpp>
#include <iostream>
#include <map>
#include <vector>
#include "spdm/SpDM.h"
using namespace simpla::data;
#define PRINT_SIZE(_TYPE_) "sizeof(" << __STRING(_TYPE_) << ") = " << sizeof(_TYPE_)
int main(int argc, char **argv) {
    simpla::data::SpDM db;
    //    db["Second"] = "The Second";
    //    db.Set("first", "The First");
    //    db["Third"]["A"] = "The Third";
    //    db["Third"]["B"]["A"] = "The Third";
    //    db["Third"]["B"]["C"] = 5;
    //
    //    db.Set("tuple3", {{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
    //    db.Set("tuple4", {{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
    //    db.Set("a",
    //           {"a"_, "not_debug"_ = false, "g"_ = {1, 2, 3, 4, 5, 5, 6, 6},
    //            "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !", "hello!"}});
    //    db.Set("/1/2"_r, 5);
    db.Set("nest", {"abc"_ = {"abc1"_ = {"def2"_ = {"abc3"_ = {"abc4"_ = "sadfsdf"}}}}});

    db["nest2"]["abc"]["abc1"]["def"]["abc"]["abc"] = "sadfsdf";

    //    std::cout << db.Get("/1/2"_r)->as<int>() << std::endl;
    //    db.Set("tuple3", {{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
    //    //    (*backend)["strlist"] = {{"abc", "def"}, {"abc", "def"}, {"abc", "def"},
    //    //    {"abc", "def"}};
    //    db.Set("tuple1", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    //    db.Set("Box", {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    //    db.Set("str_tuple", {"wa wa", "la la"});
    //
    //    db.Set("A", {1, 2, 3});
    //    db.Set("C", {{1.0, 2.0, 3.0}, {2.0}, {7.0, 9.0}});

    std::cout << db << std::endl;
}