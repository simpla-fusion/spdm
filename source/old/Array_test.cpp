//
// Created by salmon on 18-2-27.
//
#define SP_ARRAY_INITIALIZE_VALUE SP_SNaN
#include <gtest/gtest.h>

#include <complex>
#include <iostream>
#include <typeinfo>

#include "spdm/Array.h"
#include "spdm/Utility.h"
using namespace simpla;

#define EQUATION(_A, _B, _C) (-(_A + 1.0) / (_B * 2.0 - 3.0) - _C)

static constexpr double pi = 3.1415926;
template <typename T>
class TestArray : public testing::Test {
   protected:
    virtual void SetUp() {}

   public:
    typedef T type;
    typedef typename type::value_type value_type;
    static constexpr unsigned int ndim = type::ndim;
    typedef nTuple<std::ptrdiff_t, ndim> index_tuple;
};
typedef testing::Types<Array<int, 2>,                  //
                       Array<double, 3>,               //
                       Array<std::complex<double>, 1>  //
                       >
    array_type_list;

TYPED_TEST_CASE(TestArray, array_type_list);
TYPED_TEST(TestArray, initialize) {
    typename TestFixture::index_tuple dims;
    utility::Fill(dims, 3, 1);
    typename TestFixture::type vA(dims);

    EXPECT_TRUE((dims == vA.count()));
}
TYPED_TEST(TestArray, Fill) {
    typename TestFixture::index_tuple dims;
    utility::Fill(dims, 3, 1);
    typename TestFixture::type vA(dims);
    utility::Fill(vA, 0, 1);
    //    typename TestFixture::value_type count = 0;
    //    for (auto const& item : vA) {
    //        EXPECT_EQ(count, item);
    //        count += 1;
    //    }
}
TYPED_TEST(TestArray, swap) {
    typename TestFixture::index_tuple dims;
    utility::Fill(dims, 3, 1);
    typename TestFixture::type vA(dims);
    typename TestFixture::type vB(dims);
    typename TestFixture::type vC(dims);

    utility::Fill(vA, 2, 2);
    utility::Fill(vB, 1, 2);
    utility::Fill(vC, 1, 2);
    for (auto ia = begin(vA), ib = begin(vB), ea = end(vA); ia != ea; ++ia, ++ib) { EXPECT_TRUE(*ia != *ib); }
    utility::swap(vA, vB);
    for (auto ia = begin(vA), ic = begin(vC), ea = end(vA); ia != ea; ++ia, ++ic) { EXPECT_TRUE(*ia == *ic); }
}

TYPED_TEST(TestArray, assign_scalar) {
    typename TestFixture::index_tuple dims;
    utility::Fill(dims, 3, 1);
    typename TestFixture::type vA(dims);
    utility::Fill(vA, 0, 1);

    vA = pi;

    for (auto const& v : vA) { EXPECT_EQ(static_cast<typename TestFixture::value_type>(pi), v); }
}
TYPED_TEST(TestArray, assign_array) {
    typename TestFixture::index_tuple dims;
    utility::Fill(dims, 3, 1);
    typename TestFixture::type vA(dims);
    typename TestFixture::type vB(dims);

    utility::Fill(vA, 0, 1);
    utility::Fill(vB, 0, 2);

    vA = vB;

    for (auto ia = begin(vA), ib = begin(vB), ea = end(vA); ia != ea; ++ia, ++ib) { EXPECT_TRUE(*ia == *ib); }
}
TYPED_TEST(TestArray, self_assign) {
    typename TestFixture::index_tuple dims;
    utility::Fill(dims, 3, 1);
    typename TestFixture::type vA(dims);
    typename TestFixture::type vB(dims);
    typename TestFixture::type vC(dims);

    utility::Fill(vA, 0, 1);
    utility::Fill(vB, 0, 2);
    utility::Fill(vC, 0, 3);
    vA += vB;
    for (auto ia = begin(vA), ic = begin(vC), ea = end(vA); ia != ea; ++ia, ++ic) { EXPECT_TRUE(*ia == *ic); }
}
TYPED_TEST(TestArray, arithmetic) {
    typename TestFixture::index_tuple dims;
    utility::Fill(dims, 3, 1);
    typename TestFixture::type vA(dims);
    typename TestFixture::type vB(dims);
    typename TestFixture::type vC(dims);
    typename TestFixture::type vD(dims);
    utility::Fill(vA, 1, 1);
    utility::Fill(vB, 1, 2);
    utility::Fill(vC, 1, 3);
    vD = EQUATION(vA, vB, vC);
    std::size_t n = 0;
    for (auto iA = begin(vA), iB = begin(vB), iC = begin(vC), iD = begin(vD), eA = end(vA); iA != eA;
         ++iA, ++iB, ++iC, ++iD) {
        EXPECT_EQ(static_cast<typename TestFixture::value_type>(1 + 1 * n), *iA);
        EXPECT_EQ(static_cast<typename TestFixture::value_type>(1 + 2 * n), *iB);
        EXPECT_EQ(static_cast<typename TestFixture::value_type>(1 + 3 * n), *iC);
        EXPECT_EQ(static_cast<typename TestFixture::value_type>(EQUATION(*iA, *iB, *iC)), *iD);
        ++n;
    }
}
TYPED_TEST(TestArray, compare) {
    typename TestFixture::index_tuple dims;
    utility::Fill(dims, 3, 1);
    typename TestFixture::type vA(dims);
    typename TestFixture::type vB(dims);

    utility::Fill(vA, 0, 1);
    utility::Fill(vB, 1, 2);

    EXPECT_TRUE(vA == vA);
    EXPECT_TRUE(vA != vB);

    EXPECT_FALSE(vA != vA);
    EXPECT_FALSE(vA == vB);
}
TYPED_TEST(TestArray, inner_product) {
    typename TestFixture::index_tuple dims;
    utility::Fill(dims, 3, 1);
    typename TestFixture::type vA(dims);
    typename TestFixture::type vB(dims);
    utility::Fill(vA, 1, 1);
    utility::Fill(vB, 1, 2);
    auto res = inner_product(vA, vB);
    decltype(res) expect = 0;
    for (std::size_t n = 0, ne = vA.size(); n < ne; ++n) { expect += vA.get_v(n) * vB.get_v(n); }
    EXPECT_EQ(expect, res);
}
TEST(TestArray, sub_array) {
    Array<double, 2> sA{4, 6};
    utility::Fill(sA, 1, 1);
    std::cerr << sA << std::endl;
    std::cerr << sA.Slice({{1, 3, 1}, {2, 6, 2}}) << std::endl;
    std::cerr << sA.Slice({{1, 3, 1}, {2, 6, 2}}).Shift({1, 1}) << std::endl;
    utility::Fill(sA.Slice({{1, 3, 1}, {2, 6, 1}}), 0, 0);
    std::cerr << sA << std::endl;
    sA[{{0, 5, 2}, {0, 6, 2}}].FillNaN();
    std::cerr << sA << std::endl;

    VecArray<double, 2> vA;

    Reshape(vA, {4, 6});
    Initialize(vA);
    Fill(vA, 0, 1);
    std::cout << std::endl << vA << std::endl;
    std::cout << std::endl << GetByIdx(traits::as_const(vA), {1, 1}) << std::endl;
    std::cout << std::endl << Slice(vA, {{1, 3, 1}, {2, 6, 2}}) << std::endl;
    std::cout << std::endl << Shift(Slice(vA, {{1, 3, 1}, {2, 6, 2}}), {0, 0, 0}) << std::endl;

    std::cout << "The End!" << std::endl;
}
// TYPED_TEST(TestArray, get) {
//    nTuple<std::size_t, TestFixture::ndim> dims;
//    utility::Fill(dims, 3, 1);
//    typename TestFixture::type vA(dims);
//
//    utility::Fill(vA, 0, 1);
//
//    EXPECT_TRUE(vA[0] == utility::get<0>(vA));
//    EXPECT_TRUE(vA[1] == utility::get<1>(vA));
//    EXPECT_TRUE(vA[2] == utility::get<2>(vA));
//}