//
// Created by salmon on 18-2-28.
//
#include <gtest/gtest.h>
#include "spdm/TypeTraits.h"
#include "spdm/Utility.h"
using namespace sp;

TEST(spdm_traits, add_extent) {
    EXPECT_TRUE((std::is_same<int[2], traits::add_extent_t<int, 2>>::value));
    EXPECT_TRUE((std::is_same<int[2][3][4][5], traits::add_extent_t<int, 2, 3, 4, 5>>::value));
}
TEST(spdm_traits, copy_extents) {
    EXPECT_TRUE((std::is_same<int[2][3][4][5], traits::copy_extents_t<int, double[2][3][4][5]>>::value));
}

TEST(spdm_traits, is_similar) {
    EXPECT_TRUE((traits::is_similar<double, int>::value));
    EXPECT_TRUE((traits::is_similar<double[2], int[2]>::value));
    EXPECT_FALSE((traits::is_similar<double[2], double[3]>::value));
    EXPECT_TRUE((traits::is_similar<double[2][3][4][5], double[2][3][4][5]>::value));
    EXPECT_TRUE((traits::is_similar<double[2][3][4][5], int[2][3][4][5]>::value));
    EXPECT_FALSE((traits::is_similar<double[2][3][4][5], int[2][3][4][6]>::value));
    EXPECT_FALSE((traits::is_similar<double[2][3][3][5], int[2][3][4][5]>::value));
}
TEST(spdm_traits, number_of_elements) {
    EXPECT_EQ(1, (traits::number_of_elements<double>::value));

    EXPECT_EQ(2, (traits::number_of_elements<double[2]>::value));
    EXPECT_EQ(6, (traits::number_of_elements<double[2][3]>::value));
    EXPECT_EQ(24, (traits::number_of_elements<double[2][3][4]>::value));
}

TEST(spdm_traits, IsEqual) {
    double a[2][3][4];
    double b[2][3][4];
    double c[2][3][4];
    utility::Fill(a, 0, 1);
    utility::Fill(b, 0, 1);
    utility::Fill(c, 0, 2);

    EXPECT_TRUE(utility::IsEqual(a, b));
    EXPECT_FALSE(utility::IsEqual(a, c));

    //    traits::FancyPrint(std::cout, t);
}

TEST(spdm_traits, get) {
    int d1[3] = {1, 2, 3};
    EXPECT_EQ(d1[0], utility::get<0>(d1));
    EXPECT_EQ(d1[1], utility::get<1>(d1));
    EXPECT_EQ(d1[2], utility::get<2>(d1));

    int d2[3][2] = {1, 2, 3, 4, 5, 6};

    EXPECT_TRUE(utility::IsEqual(utility::get<0>(d2), d2[0]));
    EXPECT_TRUE(utility::IsEqual(utility::get<1>(d2), d2[1]));
    EXPECT_TRUE(utility::IsEqual(utility::get<2>(d2), d2[2]));
}
TEST(spdm_traits, IsEqualP) {
    double a[2][3] = {1, 2, 3, 4, 5, 6};
    double b[6] = {1, 2, 3, 4, 5, 6};
    double c[6] = {1, 2, 8, 4, 5, 6};

    EXPECT_TRUE(utility::IsEqualP(a, b));
    EXPECT_FALSE(utility::IsEqualP(a, c));

    //    traits::FancyPrint(std::cout, t);
}

TEST(spdm_traits, nested_initializer_list_traits_dims) {
    std::size_t dims[3];

    traits::nested_initializer_list_traits_dims<int, 3>(
        dims, {{{1, 2}, {1, 2}, {1, 2}}, {{1, 2}, {1, 2}}, {{1, 2}, {1, 2}}, {{1, 2}, {1, 2}}});

    EXPECT_EQ(4, dims[0]);
    EXPECT_EQ(3, dims[1]);
    EXPECT_EQ(2, dims[2]);
}