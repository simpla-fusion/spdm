//
// Created by salmon on 18-3-3.
//
#include <gtest/gtest.h>

#include <complex>
#include <iostream>
#include <typeinfo>

#include "spdm/Utility.h"
#include "spdm/nTuple.h"
using namespace simpla;

// TEST(TestnTupleTraits, is_pod) {
//    EXPECT_TRUE((std::is_pod<nTuple<int, 2>>::value));
//    EXPECT_TRUE((std::is_pod<nTuple<int, 2, 4>>::value));
//}
TEST(TestnTupleTraits, rank) {
    EXPECT_EQ(1, (traits::rank<nTuple<int, 2>>::value));
    EXPECT_EQ(3, (traits::rank<nTuple<int, 2, 2, 3>>::value));
    EXPECT_EQ(3, (traits::rank<nTuple<int, 2, 2, 3> &>::value));
    EXPECT_EQ(3, (traits::rank<nTuple<int, 2, 2, 3> &&>::value));
    EXPECT_EQ(3, (traits::rank<nTuple<int, 2, 2, 3> const>::value));
    EXPECT_EQ(3, (traits::rank<nTuple<int, 2, 2, 3> const &>::value));
}
TEST(TestnTupleTraits, extent) {
    EXPECT_EQ(2, (traits::extent<nTuple<int, 2>>::value));
    EXPECT_EQ(4, (traits::extent<nTuple<int, 3, 4, 5>, 1>::value));
    EXPECT_EQ(5, (traits::extent<nTuple<int, 3, 4, 5>, 2>::value));
    EXPECT_EQ(3, (traits::extent<nTuple<int, 3, 4, 5>>::value));
    EXPECT_EQ(3, (traits::extent<nTuple<int, 3, 4, 5> &>::value));
    EXPECT_EQ(3, (traits::extent<nTuple<int, 3, 4, 5> &&>::value));
    EXPECT_EQ(3, (traits::extent<nTuple<int, 3, 4, 5> const>::value));
    EXPECT_EQ(3, (traits::extent<nTuple<int, 3, 4, 5> const &>::value));
}

TEST(TestnTupleTraits, number_of_elements) {
    EXPECT_EQ(2, (traits::number_of_elements<nTuple<int, 2>>::value));
    EXPECT_EQ(24, (traits::number_of_elements<nTuple<int, 2, 3, 4>>::value));
    EXPECT_EQ(24, (traits::number_of_elements<nTuple<int, 2, 3, 4> &>::value));
    EXPECT_EQ(24, (traits::number_of_elements<nTuple<int, 2, 3, 4> &&>::value));
    EXPECT_EQ(24, (traits::number_of_elements<nTuple<int, 2, 3, 4> const>::value));
    EXPECT_EQ(24, (traits::number_of_elements<nTuple<int, 2, 3, 4> const &>::value));
}
TEST(TestnTupleTraits, remove_extent) {
    EXPECT_TRUE((std::is_same<nTuple<int, 3>, traits::remove_extent_t<nTuple<int, 2, 3>>>::value));
    EXPECT_TRUE((std::is_same<nTuple<int, 3>, traits::remove_extent_t<nTuple<int, 2, 3> &>>::value));
    EXPECT_TRUE((std::is_same<nTuple<int, 3>, traits::remove_extent_t<nTuple<int, 2, 3> &&>>::value));
    EXPECT_TRUE((std::is_same<nTuple<int, 3>, traits::remove_extent_t<nTuple<int, 2, 3> const>>::value));
    EXPECT_TRUE((std::is_same<nTuple<int, 3>, traits::remove_extent_t<nTuple<int, 2, 3> const &>>::value));

    EXPECT_TRUE((std::is_same<int, traits::remove_extent_t<nTuple<int, 2>>>::value));
    EXPECT_TRUE((std::is_same<int, traits::remove_all_extents_t<nTuple<int, 2, 3, 4>>>::value));
}
TEST(TestnTupleTraits, copy_extents) {
    EXPECT_TRUE((std::is_same<int[2][3][4][5], traits::copy_extents_t<int, nTuple<double, 2, 3, 4, 5>>>::value));
    EXPECT_TRUE((std::is_same<int[2][3][4][5], traits::copy_extents_t<int, nTuple<double, 2, 3, 4, 5> &>>::value));
    EXPECT_TRUE((std::is_same<int[2][3][4][5], traits::copy_extents_t<int, nTuple<double, 2, 3, 4, 5> &&>>::value));
    EXPECT_TRUE((std::is_same<int[2][3][4][5], traits::copy_extents_t<int, nTuple<double, 2, 3, 4, 5> const>>::value));
    EXPECT_TRUE(
        (std::is_same<int[2][3][4][5], traits::copy_extents_t<int, nTuple<double, 2, 3, 4, 5> const &>>::value));
}
TEST(TestnTupleTraits, is_similar) {
    EXPECT_FALSE((traits::is_similar<double[2][3][3][5], nTuple<double, 2, 3, 4, 5>>::value));
    EXPECT_TRUE((traits::is_similar<double[2][3][4][5], nTuple<double, 2, 3, 4, 5>>::value));
    EXPECT_TRUE((traits::is_similar<double[2][3][4][5], nTuple<double, 2, 3, 4, 5> &>::value));
    EXPECT_TRUE((traits::is_similar<double[2][3][4][5], nTuple<double, 2, 3, 4, 5> &&>::value));
    EXPECT_TRUE((traits::is_similar<double[2][3][4][5], nTuple<double, 2, 3, 4, 5> const>::value));
    EXPECT_TRUE((traits::is_similar<double[2][3][4][5], nTuple<double, 2, 3, 4, 5> const &>::value));

    EXPECT_FALSE((traits::is_similar<nTuple<double, 2, 3, 3, 5>, nTuple<double, 2, 3, 4, 5>>::value));
    EXPECT_TRUE((traits::is_similar<nTuple<double, 2, 3, 4, 5>, nTuple<double, 2, 3, 4, 5>>::value));
    EXPECT_TRUE((traits::is_similar<nTuple<double, 2, 3, 4, 5>, nTuple<double, 2, 3, 4, 5> &>::value));
    EXPECT_TRUE((traits::is_similar<nTuple<double, 2, 3, 4, 5>, nTuple<double, 2, 3, 4, 5> &&>::value));
    EXPECT_TRUE((traits::is_similar<nTuple<double, 2, 3, 4, 5>, nTuple<double, 2, 3, 4, 5> const>::value));
    EXPECT_TRUE((traits::is_similar<nTuple<double, 2, 3, 4, 5>, nTuple<double, 2, 3, 4, 5> const &&>::value));
}
TEST(TestnTupleTraits, initialize) {
    nTuple<double, 2> a{1, 1};
    nTuple<double, 2, 3> b{{1, 2}, {1, 2, 3}};
}