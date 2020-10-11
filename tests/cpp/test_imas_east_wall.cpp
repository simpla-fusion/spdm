#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "test_helpers.h"

#include <c++/UDA.hpp>

#define QUOTE_(X) #X
#define QUOTE(X) QUOTE_(X)
#define SHOT_NUM "55555"

/*
 * wall/description_2d/Shape_of
 */
// TEST_CASE( "Test wall description count", "[IMAS][EAST]" )
// {
// #ifdef FATCLIENT
// # include "setup.inc"
// #endif

//     uda::Client client;

//     const uda::Result& result = client.get("EAST::read(element='wall/description_2d/Shape_of', indices='', experiment='EAST', dtype=3, shot=" SHOT_NUM ", IDS_version='3')", "");

//     REQUIRE( result.errorCode() == 0 );
//     REQUIRE( result.errorMessage().empty() );

//     uda::Data* data = result.data();

//     REQUIRE( data != nullptr );
//     REQUIRE( !data->isNull() );
//     REQUIRE( data->type().name() == typeid(int).name() );

//     auto val = dynamic_cast<uda::Scalar*>(data);

//     REQUIRE( val != nullptr );
//     REQUIRE( !val->isNull() );

//     REQUIRE( val->type().name() == typeid(int).name() );
//     REQUIRE( val->as<int>() == 1 );
// }

/*
 * wall/description_2d/#/vessel/unit/#/annular/outline_inner/r
 */
TEST_CASE( "Test wall description_2d limiter unit: outline", "[IMAS][EAST]" )
{
#ifdef FATCLIENT
#  include "setup.inc"
#endif

    uda::Client client;

    const uda::Result& result = client.get("EAST::read(element='wall/description_2d/#/limiter/unit/#/outline/r', indices='1;1', experiment='EAST', dtype=7, shot=" SHOT_NUM ", IDS_version='3')", "");

    REQUIRE( result.errorCode() == 0 );
    REQUIRE( result.errorMessage().empty() );

    uda::Data* data = result.data();

    REQUIRE( data != nullptr );
    REQUIRE( !data->isNull() );
    REQUIRE( data->type().name() == typeid(double).name() );

    auto arr = dynamic_cast<uda::Array*>(data);

    std::cout<< "rank  =" << arr->shape().size()<<std::endl;
    std::cout<< "shape =" << arr->shape()[0]<< "," <<arr->shape()[1] <<std::endl;
      
    auto shape=arr->shape();
    auto vals = arr->as<double>();
    // int width=shape[0],height=shape[1];
    // for (int i=0;i<height;++i)
    // {
    //     for (int j=0;j<width;++j)
    //     {
    //         std::cout<< vals[i*width+j]<<' ';
    //     }
    //     std::cout<<std::endl;
    // }


    REQUIRE( arr != nullptr );
    REQUIRE( !arr->isNull() );

    std::vector<double> expected = {  0.485, 0.485, 0.493, 0.809, 0.809 };

    REQUIRE( arr->type().name() == typeid(double).name() );

    // auto vals = arr->as<double>();
    vals.resize(5);

    REQUIRE( vals == ApproxVector(expected) );
}
