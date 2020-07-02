//
// Created by salmon on 18-3-5.
//

#include <gtest/gtest.h>

#include <complex>
#include <iostream>
#include <typeinfo>

#include "spdm/Array.h"
#include "spdm/Utility.h"
#include "spdm/nTuple.h"

using namespace simpla;
TEST(TestnArrayTraits, traits) {
    EXPECT_EQ(3, (traits::number_of_dimensions<Array<double, 3>>::value));
    EXPECT_TRUE((std::is_same<double, traits::remove_all_dimensions_t<Array<double, 3> const>>::value));
    EXPECT_TRUE((std::is_same<Array<double, 2>, traits::remove_dimension_t<Array<double, 3> &>>::value));
    EXPECT_TRUE((std::is_same<Array<const double, 2>, traits::remove_dimension_t<Array<double, 3> const &>>::value));
    EXPECT_TRUE((std::is_same<Array<const double, 1>, traits::remove_dimension_t<Array<double, 3> const &, 2>>::value));
    EXPECT_TRUE((std::is_same<double, traits::remove_dimension_t<Array<double, 3> const &, 3>>::value));
}
TEST(TestnArrayTraits, index) {
    EXPECT_EQ(3, (traits::number_of_dimensions<Array<double, 3>>::value));
    EXPECT_TRUE((std::is_same<double, traits::remove_all_dimensions_t<Array<double, 3> const>>::value));
    EXPECT_TRUE((std::is_same<Array<double, 2>, traits::remove_dimension_t<Array<double, 3> &>>::value));
    EXPECT_TRUE((std::is_same<Array<const double, 2>, traits::remove_dimension_t<Array<double, 3> const &>>::value));
    EXPECT_TRUE((std::is_same<Array<const double, 1>, traits::remove_dimension_t<Array<double, 3> const &, 2>>::value));
    EXPECT_TRUE((std::is_same<double, traits::remove_dimension_t<Array<double, 3> const &, 3>>::value));
}