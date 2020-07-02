//
// Created by salmon on 18-2-27.
//

#include <gtest/gtest.h>

#include <complex>
#include <iostream>
#include <typeinfo>

#include "spdm/Utility.h"
#include "spdm/nTuple.h"
using namespace simpla;

//
#define EQUATION(_A, _B, _C) (-(_A + 1.0) / (_B * 2.0 - 3.0) - _C)
////#define EQUATION(_A, _B, _C) -_A* _B
static constexpr double pi = 3.1415926;
template <typename T>
class TestnTuple : public testing::Test {
   protected:
    virtual void SetUp() {}

   public:
    typedef T type;
    typedef traits::copy_extents_t<traits::remove_all_extents_t<type>, type> pod_type;

    typedef traits::remove_all_extents_t<type> value_type;
};

typedef testing::Types<nTuple<int, 3>,                  //
                       nTuple<double, 3, 4>,            //
                       nTuple<double, 3, 4, 5>,         //
                       nTuple<std::complex<double>, 3>  //,
                       // nTuple<int, 3, 4, 5, 6>,
                       // nTuple<std::complex<double>, 3, 4, 5, 6>
                       >
    tensor_type_lists;

TYPED_TEST_CASE(TestnTuple, tensor_type_lists);

TYPED_TEST(TestnTuple, Fill) {
    typename TestFixture::type vA;
    typename TestFixture::type vB;

    typename TestFixture::pod_type aA;
    typename TestFixture::pod_type aB;
    utility::Fill(vA, 0, 1);
    utility::Fill(aA, 0, 1);

    EXPECT_TRUE(utility::IsEqualP(aA, vA.m_data_));
}
TYPED_TEST(TestnTuple, compare) {
    typename TestFixture::type vA;
    typename TestFixture::type vB;

    typename TestFixture::pod_type aA;
    typename TestFixture::pod_type aB;
    utility::Fill(vA, 0, 1);
    utility::Fill(vB, 0, 2);
    utility::Fill(aA, 0, 1);
    utility::Fill(aB, 0, 2);

    EXPECT_TRUE(vA == vA);
    EXPECT_TRUE(vA != vB);
    EXPECT_TRUE(vA == aA);
    EXPECT_TRUE(vA != aB);

    EXPECT_FALSE(vA != vA);
    EXPECT_FALSE(vA == vB);
    EXPECT_FALSE(vA != aA);
    EXPECT_FALSE(vA == aB);
}

TYPED_TEST(TestnTuple, get) {
    typename TestFixture::type vA;
    utility::Fill(vA, 0, 1);
    EXPECT_TRUE(vA[0] == utility::get<0>(vA));
    EXPECT_TRUE(vA[1] == utility::get<1>(vA));
    EXPECT_TRUE(vA[2] == utility::get<2>(vA));
}

TYPED_TEST(TestnTuple, swap) {
    typename TestFixture::type vA;
    typename TestFixture::type vB;
    typename TestFixture::pod_type aA;
    typename TestFixture::pod_type aB;
    typename TestFixture::pod_type aC;

    utility::Fill(vA, 0, 1);
    utility::Fill(vB, 0, 2);

    utility::Fill(aA, 0, 1);
    utility::Fill(aB, 0, 2);
    utility::Fill(aC, 0, 1);

    utility::swap(vA, vB);
    EXPECT_TRUE(utility::IsEqualP(aB, vA.m_data_));
    utility::swap(vA, aA);
    EXPECT_TRUE(utility::IsEqualP(aC, vA.m_data_));
}
TYPED_TEST(TestnTuple, assign_scalar) {
    typename TestFixture::type vA;
    typename TestFixture::pod_type aA;
    utility::Fill(vA, 0, 1);
    vA = static_cast<typename TestFixture::value_type>(pi);
    EXPECT_TRUE(utility::IsEqualP(static_cast<typename TestFixture::value_type>(pi), vA.m_data_));
}
TYPED_TEST(TestnTuple, assign_array) {
    typename TestFixture::type vA;
    typename TestFixture::pod_type aA;
    utility::Fill(vA, 0, 1);
    utility::Fill(aA, 1, 2);
    vA = aA;
    for (std::size_t n = 0; n < traits::number_of_elements<typename TestFixture::type>::value; ++n) {
        EXPECT_EQ(static_cast<typename TestFixture::value_type>(1 + 2 * n), vA.m_data_[n]);
    }
}
TYPED_TEST(TestnTuple, self_assign) {
    typename TestFixture::type vA;
    typename TestFixture::type vB;
    typename TestFixture::pod_type aC;

    utility::Fill(vA, 0, 1);
    utility::Fill(vB, 0, 2);
    utility::Fill(aC, 0, 3);

    vB += vA;
    EXPECT_TRUE(utility::IsEqualP(aC, vB.m_data_));
}
TYPED_TEST(TestnTuple, arithmetic) {
    typename TestFixture::type vA;
    typename TestFixture::type vB;
    typename TestFixture::type vC;
    typename TestFixture::type vD;
    utility::Fill(vA, 1, 1);
    utility::Fill(vB, 1, 2);
    utility::Fill(vC, 1, 3);
    vD = EQUATION(vA, vB, vC);

    for (std::size_t n = 0; n < traits::number_of_elements<typename TestFixture::type>::value; ++n) {
        EXPECT_EQ(static_cast<typename TestFixture::value_type>(1 + 1 * n), vA.m_data_[n]);
        EXPECT_EQ(static_cast<typename TestFixture::value_type>(1 + 2 * n), vB.m_data_[n]);
        EXPECT_EQ(static_cast<typename TestFixture::value_type>(1 + 3 * n), vC.m_data_[n]);

        EXPECT_EQ(static_cast<typename TestFixture::value_type>(EQUATION(vA.m_data_[n], vB.m_data_[n], vC.m_data_[n])),
                  vD.m_data_[n]);
    }
}
TYPED_TEST(TestnTuple, expression_construct) {
    typename TestFixture::type vA;
    typename TestFixture::type vB;
    typename TestFixture::type vC;

    utility::Fill(vA, 1, 1);
    utility::Fill(vB, 1, 2);
    utility::Fill(vC, 1, 3);

    typename TestFixture::type vD = EQUATION(vA, vB, vC);
    for (std::size_t n = 0; n < traits::number_of_elements<typename TestFixture::type>::value; ++n) {
        EXPECT_EQ(static_cast<typename TestFixture::value_type>(EQUATION(vA.m_data_[n], vB.m_data_[n], vC.m_data_[n])),
                  vD.m_data_[n]);
    }
}
TYPED_TEST(TestnTuple, inner_product) {
    typename TestFixture::type vA;
    typename TestFixture::type vB;
    utility::Fill(vA, 1, 1);
    utility::Fill(vB, 1, 2);
    auto res = inner_product(vA, vB);
    decltype(res) expect = 0;
    for (std::size_t n = 0; n < traits::number_of_elements<typename TestFixture::type>::value; ++n) {
        expect += vA.m_data_[n] * vB.m_data_[n];
    }
    EXPECT_EQ(expect, res);
}
TYPED_TEST(TestnTuple, cross) {
    typename TestFixture::type vA, vB, vC, vD;

    utility::Fill(vA, 1, 1);
    utility::Fill(vB, 1, 2);
    for (std::size_t n = 0; n < traits::extent<typename TestFixture::type>::value; ++n) {
        vC[n] = vA[(n + 1) % 3] * vB[(n + 2) % 3] - vA[(n + 2) % 3] * vB[(n + 1) % 3];
    }

    vD = cross(vA, vB);

    vD -= vC;
    EXPECT_NE(0.0, std::abs(inner_product(vC, vC)));
    EXPECT_DOUBLE_EQ(0, std::abs(inner_product(vD, vD)));
}
TYPED_TEST(TestnTuple, sum) {
    typename TestFixture::type vA;
    typename TestFixture::type vB;
    utility::Fill(vA, 0, 1);
    utility::Fill(vB, 1, 2);

    auto res = sum_t(vA - vB);
    decltype(res) expect = 0;
    for (std::size_t n = 0; n < traits::number_of_elements<typename TestFixture::type>::value; ++n) {
        expect += vA.m_data_[n] - vB.m_data_[n];
    }
    EXPECT_EQ(expect, res);
}
TYPED_TEST(TestnTuple, product) {
    typename TestFixture::type vA;

    utility::Fill(vA, 1, 1);
    auto res = product_t(vA);
    decltype(res) expect = 1;

    for (std::size_t n = 0; n < traits::number_of_elements<typename TestFixture::type>::value; ++n) {
        expect *= vA.m_data_[n];
    }

    EXPECT_DOUBLE_EQ(std::abs(res), std::abs(expect));
}
TYPED_TEST(TestnTuple, fma) {
    typename TestFixture::type vA;
    typename TestFixture::type vB;
    typename TestFixture::type vC;
    typename TestFixture::type vD;
    typename TestFixture::type vE;

    utility::Fill(vA, 0, 1);
    utility::Fill(vB, 2, 3);
    utility::Fill(vC, 3, 2);
    vD = fma(vA, vB, vC);
    auto res = sum_t(fma(vA, vB - vA, vC));
    decltype(res) expect = 0;
    for (std::size_t n = 0; n < traits::number_of_elements<typename TestFixture::type>::value; ++n) {
        EXPECT_EQ(vD.m_data_[n], vA.m_data_[n] * vB.m_data_[n] + vC.m_data_[n]);
        expect += vA.m_data_[n] * (vB.m_data_[n] - vA.m_data_[n]) + vC.m_data_[n];
    }
    EXPECT_EQ(expect, res);
}
TEST(TestnTuple, initialize) {
    nTuple<double, 2> a{1, 1};
    nTuple<double, 2, 3> b{{1, 2}, {1, 2, 3}};
    std::cerr << b << std::endl;
}