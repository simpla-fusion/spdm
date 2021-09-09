//
// Created by salmon on 18-1-26.
//

#ifndef SIMPLA_SPDM_TEST_H
#define SIMPLA_SPDM_TEST_H
#include "SpDM.h"
#include <gtest/gtest.h>
using namespace sp;

template <typename TDataModel>
class TestDataModel : public testing::Test
{
public:
    typedef typename TDataModel::object_type object_type;
    typedef typename TDataModel::value_type value_type;
    typedef typename TDataModel::key_type key_type;
    typedef typename TDataModel::array_type array_type;
    typedef typename TDataModel::number_type number_type;
    typedef typename TDataModel::string_type string_type;
};
static constexpr double pi = 3.1415926535897932384626433;

TYPED_TEST_CASE_P(TestDataModel);
TYPED_TEST_P(TestDataModel, AsNumber)
{
    typename TestFixture::number_type db;

    db.Set(5);
    EXPECT_EQ(5, db.template as<int>());
    EXPECT_STREQ("5", db.template as<std::string>().c_str());
    auto pi_str = std::to_string(pi);
    db.Set(pi);
    EXPECT_DOUBLE_EQ(pi, db.template as<double>());
    EXPECT_STREQ(pi_str.c_str(), db.template as<std::string>().c_str());

    db = pi;
    EXPECT_EQ(pi, db.template as<double>());

    db = true;
    EXPECT_TRUE(db);

    db = false;
    EXPECT_FALSE(db);

    db = 5;
    EXPECT_TRUE((db == 5));
}
TYPED_TEST_P(TestDataModel, AsString)
{
    static const char longstr[] = "This is a string! This is a string!This is a string!This is a string!";
    static const char shortstr[] = "str";
    typename TestFixture::string_type db(longstr);
    EXPECT_STREQ(longstr, db.c_str());

    db.Set(shortstr);
    EXPECT_STREQ(shortstr, db.c_str());

    db = longstr;
    EXPECT_STREQ(longstr, db.c_str());
    typename TestFixture::string_type db1(db);
    EXPECT_STREQ(longstr, db1.c_str());

    db = shortstr;
    EXPECT_STREQ(shortstr, db.c_str());

    db.Set("5");
    EXPECT_EQ(5, db.template as<int>());

    double d = std::strtod(std::to_string(pi).c_str(), nullptr);
    auto dstr = std::to_string(d);
    db.Set(dstr);
    EXPECT_DOUBLE_EQ(d, db.template as<double>());

    db.Set(5);
    EXPECT_EQ(5, db.template as<int>());
    EXPECT_STREQ("5", db.template as<std::string>().c_str());
    auto pi_str = std::to_string(pi);
    double pi_d = atof(pi_str.c_str());
    db.Set(pi);
    EXPECT_DOUBLE_EQ(pi_d, db.template as<double>());
    EXPECT_STREQ(pi_str.c_str(), db.template as<std::string>().c_str());

    db = pi;
    EXPECT_EQ(pi_d, db.template as<double>());
}
TYPED_TEST_P(TestDataModel, AsNumberTensor)
{
    typename TestFixture::number_type db;

    size_t dims[4];
    db.Set({1, 1, 2});
    EXPECT_TRUE(db.isTensor());
    EXPECT_EQ(1, db.GetRank());
    EXPECT_EQ(3, db.element_size());
    db.GetDimensions(dims);
    EXPECT_EQ(3, dims[0]);

    db.Set({{1, 2, 3}, {4, 5, 6}});
    EXPECT_TRUE(db.isTensor());
    EXPECT_EQ(2, db.GetDimensions(dims));
    EXPECT_EQ(6, db.element_size());
    EXPECT_EQ(2, dims[0]);
    EXPECT_EQ(3, dims[1]);

    db.Set({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    EXPECT_EQ(3, db.GetDimensions(dims));
    EXPECT_EQ(2, dims[0]);
    EXPECT_EQ(2, dims[1]);
    EXPECT_EQ(2, dims[2]);
    int* d = db.template asTensor<int>();
    EXPECT_EQ(1, d[0]);
    EXPECT_EQ(2, d[1]);
    EXPECT_EQ(3, d[2]);
    EXPECT_EQ(4, d[3]);
    EXPECT_EQ(5, d[4]);
    EXPECT_EQ(6, d[5]);
    EXPECT_EQ(7, d[6]);
    EXPECT_EQ(8, d[7]);

    db.Set({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
    EXPECT_EQ(2, db.GetDimensions(dims));
    EXPECT_EQ(4, dims[0]);
    EXPECT_EQ(2, dims[1]);
    d = db.template asTensor<int>();
    EXPECT_EQ(1, d[0]);
    EXPECT_EQ(2, d[1]);
    EXPECT_EQ(3, d[2]);
    EXPECT_EQ(4, d[3]);
    EXPECT_EQ(5, d[4]);
    EXPECT_EQ(6, d[5]);
    EXPECT_EQ(7, d[6]);
    EXPECT_EQ(8, d[7]);

    std::cerr << db.template as<nTuple<int, 2, 4>>() << std::endl;
    db.template as<nTupleView<int, 2, 4>>() = 3;
    std::cerr << db.template as<nTuple<int, 2, 4>>() << std::endl;
}
TYPED_TEST_P(TestDataModel, AsObject)
{
    typename TestFixture::object_type db;

    db["First"] = 5;
    db["Second"] = "This is second!";

    //    EXPECT_TRUE(db.isObject());
    EXPECT_TRUE(db.Find("First"));
    EXPECT_TRUE(db.Find("Second"));
    EXPECT_FALSE(db.Find("Third"));
    EXPECT_EQ(2, db.size());
    EXPECT_EQ(5, db["First"].template as<int>());

    EXPECT_STREQ("This is second!", db["Second"]->asString().c_str());

    db["First"].Add(12);
    db["First"].Add(pi);
    EXPECT_EQ(3, db["First"]->size());
    EXPECT_DOUBLE_EQ(pi, db["First"][2]->template as<double>());
}
TYPED_TEST_P(TestDataModel, AsArray)
{
    typename TestFixture::array_type db;

    db.Add(1);
    db.Add(2);
    db.Add(3);
    db.Add(4);
    db.Add(5);
    EXPECT_EQ(5, db.size());
    typename TestFixture::array_type db2;

    db2.Add(4);
    db2.Add(5);
    db2.Add(6);

    db.Merge(db2);
    EXPECT_EQ(8, db.size());
}
TYPED_TEST_P(TestDataModel, AsObjectValue)
{
    typename TestFixture::value_type db;

    db.InsertOrAssign("a", 5);
    db.InsertOrAssign("aa", "This is second!");
    EXPECT_TRUE(db.Find("a"));
    EXPECT_TRUE(db.Find("aa"));
    EXPECT_FALSE(db.Find("aab"));
    EXPECT_EQ(2, db.size());
    EXPECT_EQ(5, db.at("a").template as<int>());
    EXPECT_STREQ("This is second!", db.at("aa").template as<std::string>().c_str());

    db.Delete("a");
    EXPECT_EQ(1, db.size());
    EXPECT_FALSE(db.Find("a"));
    EXPECT_TRUE(db.Find("aa"));
}
TYPED_TEST_P(TestDataModel, AsArrayValue)
{
    typename TestFixture::value_type db;

    db.Add("balala0");
    db.Add("balala1");
    db.Add("balala2");
    db.Add("balala3");
    db.Add("balala4");
    EXPECT_EQ(5, db.size());
    EXPECT_STREQ("balala2", db.at(2).template as<std::string>().c_str());
    db.Delete(3);

    EXPECT_STREQ("balala4", db.at(3).template as<std::string>().c_str());
    db.InsertOrAssign(3, "aqwewerwew");
    EXPECT_STREQ("aqwewerwew", db.at(3).template as<std::string>().c_str());
}
TYPED_TEST_P(TestDataModel, AsTupleValue)
{
    typename TestFixture::value_type db;

    auto t0 = std::make_tuple(1, 2.0, pi, std::string("hello world"));
    db.Set(t0);
    EXPECT_TRUE(db.isArray());
    EXPECT_EQ(std::get<0>(t0), db[0].template as<int>());
    EXPECT_DOUBLE_EQ(std::get<1>(t0), db[1].template as<double>());
    EXPECT_DOUBLE_EQ(std::get<2>(t0), db[2].template as<double>());
    EXPECT_STREQ(std::get<3>(t0).c_str(), db[3].template as<std::string>().c_str());
    auto t1 = db.template as<std::tuple<int, double, double, std::string>>();
    EXPECT_EQ(std::get<0>(t0), std::get<0>(t1));
    EXPECT_DOUBLE_EQ(std::get<1>(t0), std::get<1>(t1));
    EXPECT_DOUBLE_EQ(std::get<2>(t0), std::get<2>(t1));
    EXPECT_STREQ(std::get<3>(t0).c_str(), std::get<3>(t1).c_str());
}
TYPED_TEST_P(TestDataModel, InitializerList)
{
    typename TestFixture::object_type db =
        ("abc"_ = ("abc1"_ = ("def2"_ = ("abc3"_ = ("abc4"_ = "sadfsdf", "abc5"_ = pi)))));
    EXPECT_TRUE(db.Find("abc"));
    EXPECT_TRUE(db.at("abc").Find("abc1"));
    EXPECT_TRUE(db.at("abc").at("abc1").Find("def2"));
    EXPECT_TRUE(db.at("abc").at("abc1").at("def2").Find("abc3"));
    EXPECT_TRUE(db.at("abc").at("abc1").at("def2").at("abc3").Find("abc4"));
    EXPECT_STREQ("sadfsdf",
                 db.at("abc").at("abc1").at("def2").at("abc3").at("abc4").template as<std::string>().c_str());
}
TYPED_TEST_P(TestDataModel, AsReferenceReturn)
{
    typename TestFixture::value_type db;
    db["Third"]["4"]["five"] = pi;
    EXPECT_EQ(1, db.size());
    EXPECT_TRUE(db.Find("Third"));
    EXPECT_TRUE(db.Find("Third")->Find("4")->Find("five"));
    EXPECT_DOUBLE_EQ(pi, db["Third"]["4"]["five"].template as<double>());
    db["Third"]["4"]["five"].Add(2.0 * pi);
    EXPECT_EQ(2, db["Third"]["4"]["five"].size());
    EXPECT_DOUBLE_EQ(2.0 * pi, db["Third"]["4"]["five"][1].template as<double>());
}
TYPED_TEST_P(TestDataModel, RecursivePath)
{
    typename TestFixture::value_type db;
    db.Insert("a/b/c/d")->Add(5);
    db.Insert("a/b/c/d")->Add(6);

    EXPECT_TRUE(db.Find("a/b/c/d"));
    EXPECT_EQ(2, db.size("a/b/c/d"));

    EXPECT_EQ(5, (db.at("a/b/c/d/0").template as<int>()));

    db = ("abc"_ = ("abc1"_ = ("def2"_ = ("abc3"_ = ("abc4"_ = "sadfsdf", "abc6"_ = {1, 2, 3, 4, 5})))));

    db["abc7"] = {1, 2, 3, 4, 5};
    EXPECT_EQ(2, db.size("/abc/abc1/def2/abc3"));
    EXPECT_TRUE(db.Find("abc/abc1/def2/abc3/abc4"));
    EXPECT_STREQ("sadfsdf", (db.at("abc/abc1/def2/abc3/abc4").template as<std::string>().c_str()));
    db.InsertOrAssign("abc/abc1/def2/abc3/abc4", 5);
    EXPECT_EQ(2, db.size("/abc/abc1/def2/abc3"));
    EXPECT_TRUE(db.Find("abc/abc1/def2/abc3/abc4"));
    EXPECT_EQ(5, (db.at("abc/abc1/def2/abc3/abc4").template as<int>()));
    EXPECT_TRUE(db.Find("abc/abc1/def2/abc3")->isObject());
    db.InsertOrAssign("abc/abc1/def2/abc3/abc5", pi);
    EXPECT_TRUE(db.Find("abc/abc1/def2/abc3/abc5"));
    EXPECT_DOUBLE_EQ(pi, (db.at("abc/abc1/def2/abc3/abc5").template as<double>()));
    EXPECT_EQ(3, db.size("abc/abc1/def2/abc3"));
    db.Delete("/abc/abc1/def2/abc3/abc5");
    EXPECT_EQ(2, db.size("/abc/abc1/def2/abc3"));
    EXPECT_TRUE(db.Find("/abc/abc1/def2/abc3/abc4"));
    EXPECT_FALSE(db.Find("/abc/abc1/def2/abc3/abc5"));
}
TYPED_TEST_P(TestDataModel, AsReference)
{
    typename TestFixture::object_type db;

    typename TestFixture::value_type value;

    value.Set(&db, kIsReference);
    value["Third"]["4"]["five"] = pi;

    EXPECT_EQ(1, db.size());
    EXPECT_TRUE(db.Find("Third"));
    EXPECT_TRUE(db.Find("Third")->Find("4")->Find("five"));
    EXPECT_DOUBLE_EQ(pi, db["Third"]["4"]["five"].template as<double>());
    db["Third"]["4"]["five"].Add(2.0 * pi);
    EXPECT_EQ(2, value["Third"]["4"]["five"].size());
    EXPECT_DOUBLE_EQ(2.0 * pi, value["Third"]["4"]["five"][1].template as<double>());

    value.reset();

    EXPECT_TRUE(value.isNull());

    EXPECT_EQ(1, db.size());
    EXPECT_TRUE(db.Find("Third"));
    EXPECT_TRUE(db.Find("Third")->Find("4")->Find("five"));

    EXPECT_EQ(2, db["Third"]["4"]["five"].size());
    EXPECT_DOUBLE_EQ(pi, db["Third"]["4"]["five"][0].template as<double>());
    EXPECT_DOUBLE_EQ(2.0 * pi, db["Third"]["4"]["five"][1].template as<double>());
}
TYPED_TEST_P(TestDataModel, CheckEqual)
{
    typename TestFixture::value_type db0;
    typename TestFixture::value_type db1;
    db0 = ("a"_ = {{3.0, 1.0}, {1.0, 2.0}, {3.0, 5.0}},
           ("abc"_ = ("abc1"_ = ("def2"_ = ("abc6"_ = true, "abc7"_ = false, "abc8"_ = 1,
                                            "abc3"_ = ("abc4"_ = "sadfsdf", "abc6"_ = {1, 2, 3, 4, 5}))))));
    db1 = ("a"_ = {{3.0, 1.0}, {1.0, 2.0}, {3.0, 5.0}},
           ("abc"_ = ("abc1"_ = ("def2"_ = ("abc6"_ = true, "abc7"_ = false, "abc8"_ = 1,
                                            "abc3"_ = ("abc4"_ = "sadfsdf", "abc6"_ = {1, 2, 3, 4, 5}))))));
    EXPECT_TRUE(db0 == db1);
    EXPECT_FALSE(db0 != db1);
    db0["t_array"].Add(5);
    db0["t_array"].Add(6);
    db0["t_array"].Add("hello world!");
    db0["t_array"].Add("a"_ = 3.1416926);

    EXPECT_FALSE(db0 == db1);
    EXPECT_TRUE(db0 != db1);
    db1["t_array"].Add(5);
    db1["t_array"].Add(6);
    db1["t_array"].Add("hello world!");
    db1["t_array"].Add("a"_ = 3.1416926);
    EXPECT_TRUE(db0 == db1);
    EXPECT_FALSE(db0 != db1);
}
TYPED_TEST_P(TestDataModel, Property)
{
    struct Foo : public TestFixture::object_type
    {
        typedef typename TestFixture::object_type::object_type object_type;
        SP_PROPERTY(int, Data) = 123;
        SP_PROPERTY(sp::nTuple<int, 3>, VData) = {4, 5, 6};
        SP_PROPERTY_STR(Name);
    };
    Foo foo;
    EXPECT_EQ(foo.GetData(), foo["Data"].template as<int>());
    EXPECT_TRUE(foo.GetVData() == (foo["VData"].template as<sp::nTuple<int, 3>>()));
    foo.SetName("Hello");
    foo["Data"]->Set(128);
    EXPECT_EQ(128, foo.GetData());
    sp::nTuple<int, 3> v{7, 8, 9};
    foo["VData"]->Set(v);
    EXPECT_TRUE(v == (foo["VData"].template as<sp::nTuple<int, 3>>()));
    EXPECT_TRUE((v == foo.GetVData()));
    EXPECT_TRUE((v == foo.m_VData_));
    sp::nTuple<int, 3> v2{9, 8, 7};
    foo.m_VData_ = {9, 8, 7};
    EXPECT_TRUE(v2 == (foo["VData"].template as<sp::nTuple<int, 3>>()));
    EXPECT_TRUE((v2 == foo.GetVData()));
    EXPECT_TRUE((v2 == foo.m_VData_));
}

REGISTER_TYPED_TEST_CASE_P(TestDataModel, AsNumber, AsString, AsNumberTensor, AsObject, AsArray, AsObjectValue,
                           AsArrayValue, AsTupleValue, InitializerList, AsReferenceReturn, RecursivePath, AsReference,
                           CheckEqual, Property);

#endif // SIMPLA_SPDM_TEST_H

//#include <SpDM.h>
//#include <gtest/gtest.h>
//#include <complex>
//#include <iostream>
//
// using namespace sp;
// using namespace sp::ptr;
//
// class SpDMTest : public testing::Test {
// protected:
//    void SetUp() {}
//    void TearDown() { db.Destroy(); }
//
// public:
//    virtual ~SpDMTest() {}
//    SpDM db;
//};
//
// TEST_F(SpDMTest, light_data_sigle_value) {
//    db.Set("CartesianGeometry", "hello world!");
//    db.Set("b", 5.0);
//
//    EXPECT_STREQ(db->Get("CartesianGeometry")->as<std::string>().c_str(), "hello world!");
//    EXPECT_DOUBLE_EQ(db->Get("b")->as<double>(), 5);
//    EXPECT_EQ(db->Get("b")->as<unsigned int>(), 5);
//
//    EXPECT_EQ(db.m_size_(), 2);
//    db.Add("c", 5.0);
//    db.Add("c", 5.0);
//    db.Add("c", 5.0);
//    db.Add("c", 5.0);
//    std::cout << db << std::endl;
//}
//
// TEST_F(SpDMTest, light_data_Set_ntuple) {
//    db.Set("b", 5.0);
//
//    //    db.Set("tuple10", {{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
//    //    db.Set("tuple1", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
//    //    db.Set("Box", {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
//    //    db.Set("str_tuple", {"wa wa", "la la"});
//    //
//    //    db.Set("A", {1, 2, 3});
//    //    db.Set("C", {{1.0, 2.0, 3.0}, {2.0}, {7.0, 9.0}});
//
//    //    std::cout << db << std::endl;
//
//    //  EXPECT_EQ((db->Get("tuple1")->as<nTuple<Real, 6>>(),
//    //            (nTuple<Real, 6>{1, 2, 3, 4, 5, 6}));
//    //  EXPECT_EQ((db->Get("Box")->as<nTuple<Real, 2, 3>>()),
//    //            (nTuple<Real, 2, 3>{{1, 2, 3}, {4, 5, 6}}));
//}
// TEST_F(SpDMTest, light_data_Add) {
//    db.Add({0, 5, 3, 4});
//    db.Add({1, 5, 3, 4});
//
//    std::cout << db << std::endl;
//
//    //    EXPECT_EQ((db->Get("a/1")->as<nTuple<int, 4>>()), (nTuple<int, 4>{1, 5, 3, 4}));
//    //    EXPECT_EQ((db->Get("a")->as<nTuple<int, 2, 4>>), (nTuple<int, 2, 4>{{0, 5, 3, 4}, {1, 5, 3, 4}}));
//}
//
// TEST_F(SpDMTest, light_data_multilevel) {
//    db.Set("a/b/sub/1/2/3/4/d", 5.0);
//    db.Set("/1/2/3/4/d", 5);
//    //    std::cout << db << std::endl;
//
//    EXPECT_DOUBLE_EQ(db->Get("a/b/sub/1/2/3/4/d")->as<Real>(), 5);
//    EXPECT_EQ((db->Get("/1/2/3/4/d"))->as<int>(), 5);
//}
// TEST_F(SpDMTest, light_data_keyvalue) {
//    //    db.Set("i", {"default"_, "abc"_ = 1, "abc"_ = "def", "abc"_ = 2, "abc"_ = "sadfsdf"});
//    db.Set("a",
//           {"a"_, "not_debug"_ = false, "g"_ = {1, 2, 3, 4, 5, 5, 6, 6},
//                   "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !", "hello!"}});
//    //    (*backend)["h"] = {{"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"},
//    //    {"abc"_ = "def"}};
//    db.Set("nest", {"abc"_ = {"abc1"_ = {"def"_ = {"abc"_ = {"abc"_ = "sadfsdf"}}}}});
//    std::cout << db << std::endl;
//    EXPECT_TRUE(db->Get("a/a"));
//    EXPECT_FALSE(*db->Get("a/not_debug"));
//}
//
////
//// TEST_F(DataObject, samrai) {
////    logger::set_stdout_level(1000);
////
////    LOGGER << "Registered SpDM: " <<
////    GLOBAL_DATA_BACKEND_FACTORY.GetBackendList() << std::endl;
////    DataTable backend("samrai://");
////    //   backend->SetEntity("f", {1, 2, 3, 4, 5, 56, 6, 6});
////    //   backend->SetEntity("/d/e/f", "Just atest");
////    //   backend->SetEntity("/d/e/g", {"a"_ = "la la land", "b"_ = 1235.5});
////    //   backend->SetEntity("/d/e/e", 1.23456);
////   backend->SetEntity("box", {{1, 2, 3}, {4, 5, 6}});
////    LOGGER << *backend.database() << std::endl;
////    LOGGER << "box  = " <<backend->GetEntity<std::tuple<nTuple<int, 3>,
////    nTuple<int, 3>>>("box") << std::endl;
////
////}
//// TEST(DataTable, lua) {
////    auto db = DataEntry::New("lua://");
////    db->Parse(
////        "PI = 3.141592653589793, \n "
////        "c = 299792458.0, -- m/s\n"
////        "qe = 1.60217656e-19, -- C\n"
////        "me = 9.10938291e-31, --kg\n"
////        "mp = 1.672621777e-27, --kg\n"
////        "mp_me = 1836.15267245, --\n"
////        "KeV = 1.1604e7, -- K\n"
////        "Tesla = 1.0, -- Tesla\n"
////        "TWOPI =  math.pi * 2,\n"
////        "k_B = 1.3806488e-23, --Boltzmann_constant\n"
////        "epsilon0 = 8.8542e-12,\n"
////        "AAA = { c =  3 , d = { c = \"3\", e = { 1, 3, 4, 5 } } },\n"
////        "CCC = { 1, 3, 4, 5 },\n"
////        "Box={{1,2,3},{3,4,5}},\n"
////        "tuple3={{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}},\n"
////        " nest = { abc= {abc1= {def = {abc = {abc= \"sadfsdf\"}}}}}");
////    db->Set("Box2", {{0, 5, 3}, {1, 5, 3}});
////    MESSAGE << "lua:// " << (*db) << std::endl;
////    //    MESSAGE << "Box " << (*backend)["Context/Box"]->as<nTuple<int, 2,
////    3>>() << std::endl;
////    EXPECT_EQ(db->GetValue<int>("AAA/c"), 3);
////    EXPECT_EQ((db->GetValue<nTuple<int, 4>>("/CCC")), (nTuple<int, 4>{1, 3, 4,
////    5}));
////
////    db->Flush();
////    //
////    //    EXPECT_DOUBLE_EQ((*backend)["/Context/c"]->as<double>(), 299792458);
////
////    //   backend->SetEntity("box", {{1, 2, 3}, {4, 5, 6}});
////    //    LOGGER << "box  = " <<backend->GetEntity<std::tuple<nTuple<int, 3>,
////    nTuple<int, 3>>>("box") << std::endl;
////}
