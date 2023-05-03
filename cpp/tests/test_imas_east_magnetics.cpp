#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "test_helpers.h"
#include <c++/UDA.hpp>

#define QUOTE_(X) #X
#define QUOTE(X) QUOTE_(X)
#define SHOT_NUM "55555"

/*
 * magnetics/bpol_probe/#/position
 */
// TEST_CASE( "Test magnetics/bpol_probe//Shape_of", "[IMAS][EAST]" )
// {
// #ifdef FATCLIENT
// # include "setup.inc"
// #endif

//     uda::Client client;

//     const uda::Result& result = client.get("EAST::read(element='magnetics/bpol_probe/Shape_of', indices='', experiment='EAST', dtype=3, shot=" SHOT_NUM ", IDS_version='3')", "");

//     REQUIRE( result.errorCode() == 0 );
//     REQUIRE( result.errorMessage().empty() );

//     uda::Data* data = result.data();

//     REQUIRE( data != nullptr );
//     REQUIRE( !data->isNull() );
//     REQUIRE( data->type().name() == typeid(int).name() );

//     auto val = dynamic_cast<uda::Scalar*>(data);

//     REQUIRE( val != nullptr );
//     REQUIRE( !val->isNull() );

//     REQUIRE( val->as<int>() == 76 );
//     REQUIRE( val->type().name() == typeid(int).name() );

// }
// TEST_CASE( "Test magnetics bpol_probe position r ", "[IMAS][EAST]" )
// {
// #ifdef FATCLIENT
// #  include "setup.inc"
// #endif

//     uda::Client client;

//     const uda::Result& result = client.get("EAST::read(element='magnetics/bpol_probe/#/position/r', indices='1', experiment='EAST', dtype=7, shot=" SHOT_NUM ", IDS_version='3')", "");

//     REQUIRE( result.errorCode() == 0 );
//     REQUIRE( result.errorMessage().empty() );

//     uda::Data* data = result.data();

//     REQUIRE( data != nullptr );
//     REQUIRE( !data->isNull() );
//     REQUIRE( data->type().name() == typeid(double).name() );

//     auto arr = dynamic_cast<uda::Scalar*>(data);
//      REQUIRE( arr != nullptr );
//     REQUIRE( !arr->isNull() );
//     REQUIRE( arr->type().name() == typeid(double).name() );
//     REQUIRE( arr->as<double>() == 1.289 );
// }
// TEST_CASE( "Test magnetics bpol_probe position z ", "[IMAS][EAST]" )
// {
// #ifdef FATCLIENT
// #  include "setup.inc"
// #endif

//     uda::Client client;

//     const uda::Result& result = client.get("EAST::read(element='magnetics/bpol_probe/#/position/z', indices='1', experiment='EAST', dtype=7, shot=" SHOT_NUM ", IDS_version='3')", "");

//     REQUIRE( result.errorCode() == 0 );
//     REQUIRE( result.errorMessage().empty() );

//     uda::Data* data = result.data();

//     REQUIRE( data != nullptr );
//     REQUIRE( !data->isNull() );
//     REQUIRE( data->type().name() == typeid(double).name() );

//     auto arr = dynamic_cast<uda::Scalar*>(data);
//      REQUIRE( arr != nullptr );
//     REQUIRE( !arr->isNull() );
//     REQUIRE( arr->type().name() == typeid(double).name() );
//     REQUIRE( arr->as<double>() == 0.2495 );
// }
// TEST_CASE( "Test magnetics bpol_probe position poloidal_angle ", "[IMAS][EAST]" )
// {
// #ifdef FATCLIENT
// #  include "setup.inc"
// #endif

//     uda::Client client;

//     const uda::Result& result = client.get("EAST::read(element='magnetics/bpol_probe/#/poloidal_angle', indices='1', experiment='EAST', dtype=7, shot=" SHOT_NUM ", IDS_version='3')", "");

//     REQUIRE( result.errorCode() == 0 );
//     REQUIRE( result.errorMessage().empty() );

//     uda::Data* data = result.data();

//     REQUIRE( data != nullptr );
//     REQUIRE( !data->isNull() );
//     REQUIRE( data->type().name() == typeid(double).name() );

//     auto arr = dynamic_cast<uda::Scalar*>(data);
//      REQUIRE( arr != nullptr );
//     REQUIRE( !arr->isNull() );
//     REQUIRE( arr->type().name() == typeid(double).name() );
//     REQUIRE( arr->as<double>() == 88.0459 );
// }

TEST_CASE("Test magnetics/flux_loop//Shape_of", "[IMAS][EAST]")
{
// #ifdef FATCLIENT
#include "setup.inc"
// #endif

    uda::Client client;
    std::cout<<"hello world"<<std::endl;

    const uda::Result &result = client.get("EAST::read(element='magnetics/flux_loop/Shape_of', indices='', experiment='EAST', dtype=3, shot=" SHOT_NUM ", IDS_version='3')", "");
    std::cout<<"hello world"<<std::endl;
    REQUIRE(result.errorCode() == 0);
    REQUIRE(result.errorMessage().empty());

    uda::Data *data = result.data();

    REQUIRE(data != nullptr);
    REQUIRE(!data->isNull());
    REQUIRE(data->type().name() == typeid(int).name());

    auto val = dynamic_cast<uda::Scalar *>(data);

    REQUIRE(val != nullptr);
    REQUIRE(!val->isNull());

    REQUIRE(val->as<int>() == 4);
    REQUIRE(val->type().name() == typeid(int).name());
}
TEST_CASE("Test magnetics/flux_loop/position/Shape_of", "[IMAS][EAST]")
{
// #ifdef FATCLIENT
#include "setup.inc"
// #endif

    uda::Client client;

    const uda::Result &result = client.get("EAST::read(element='magnetics/flux_loop/#/position/Shape_of', indices='1', experiment='EAST', dtype=3, shot=" SHOT_NUM ", IDS_version='3')", "");

    REQUIRE(result.errorCode() == 0);
    REQUIRE(result.errorMessage().empty());

    uda::Data *data = result.data();

    REQUIRE(data != nullptr);
    REQUIRE(!data->isNull());
    REQUIRE(data->type().name() == typeid(int).name());

    auto val = dynamic_cast<uda::Scalar *>(data);

    REQUIRE(val != nullptr);
    REQUIRE(!val->isNull());

    REQUIRE(val->as<int>() == 2);
    REQUIRE(val->type().name() == typeid(int).name());
}
TEST_CASE("Test magnetics flux_loop position r ", "[IMAS][EAST]")
{
// #ifdef FATCLIENT
#include "setup.inc"
// #endif

    uda::Client client;

    const uda::Result &result = client.get("EAST::read(element='magnetics/flux_loop/#/position/r', indices='1', experiment='EAST', dtype=7, shot=" SHOT_NUM ", IDS_version='3')", "");

    REQUIRE(result.errorCode() == 0);
    REQUIRE(result.errorMessage().empty());

    uda::Data *data = result.data();

    REQUIRE(data != nullptr);
    REQUIRE(!data->isNull());
    REQUIRE(data->type().name() == typeid(double).name());

    auto arr = dynamic_cast<uda::Scalar *>(data);
    REQUIRE(arr != nullptr);
    REQUIRE(!arr->isNull());
    REQUIRE(arr->type().name() == typeid(double).name());
    REQUIRE(arr->as<double>() == 1.2707);
}

TEST_CASE("Test magnetics flux_loop.field.data", "[IMAS][EAST]")
{
#ifdef FATCLIENT
#include "setup.inc"
#endif

    uda::Client client;
    int count = 0;

    for (int i = 0; i < 76; ++i)
    {
        const uda::Result &result = client.get("EAST::read(element='magnetics/bpol_probe/#/field/data', indices='" + std::to_string(i) + "', experiment='EAST', dtype=7, shot=" SHOT_NUM ", IDS_version='3')", "");
        if (result.data()->isNull())
        {
            std::cout << "EAST::read(element='magnetics/bpol_probe/#/field/data', indices='" << std::to_string(i) << "', experiment='EAST', dtype=7, shot=" << SHOT_NUM << ", IDS_version='3')" << std::endl;
            ++count;
        }
    }

    REQUIRE(count == 0);
}