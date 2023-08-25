#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../east_plugin.h"
#include <clientserver/initStructs.h>
#include <clientserver/udaTypes.h>
#include <structures/struct.h>

IDAM_PLUGIN_INTERFACE generate_plugin_interface(const std::string &function, const std::vector<std::pair<std::string, std::string>> &arguments)
{
    IDAM_PLUGIN_INTERFACE interface = {0};

    interface.dbgout = stdout;
    interface.errout = stderr;

    interface.data_block = new DATA_BLOCK;
    initDataBlock(interface.data_block);

    interface.request_block = new REQUEST_BLOCK;
    initRequestBlock(interface.request_block);

    strcpy(interface.request_block->function, function.c_str());

    initNameValueList(&interface.request_block->nameValueList);
    interface.request_block->nameValueList.listSize = (int)arguments.size();
    interface.request_block->nameValueList.pairCount = (int)arguments.size();
    interface.request_block->nameValueList.nameValue = (NAMEVALUE *)calloc(arguments.size(), sizeof(NAMEVALUE));
    int i = 0;
    for (auto &pair : arguments)
    {
        NAMEVALUE *nv = &interface.request_block->nameValueList.nameValue[i];
        nv->pair = strdup((pair.first + "=" + pair.second).c_str());
        nv->name = strdup(pair.first.c_str());
        nv->value = strdup(pair.second.c_str());
        ++i;
    }

    interface.userdefinedtypelist = new USERDEFINEDTYPELIST;
    initUserDefinedTypeList(interface.userdefinedtypelist);

    interface.logmalloclist = new LOGMALLOCLIST;
    initLogMallocList(interface.logmalloclist);

    return interface;
}

TEST_CASE("Test homogeneous_time", "[IMAS][EAST]")
{
    // #ifdef FATCLIENT
    setenv("UDA_EAST_MAPPING_FILE_DIRECTORY", "/workspaces/uda_plugins/source/east/tests/../mappings", 1);
    // #endif

    // auto user = getenv("USER");

    IDAM_PLUGIN_INTERFACE interface = generate_plugin_interface("read", {{"shot", "1000"},
                                                                         {"run", "0"},
                                                                         {"user", "fydev"},
                                                                         {"experiment", "EAST"},
                                                                         {"version", "3"},
                                                                         {"element", "/magnetics/flux_loop/#/flux/data"},
                                                                         {"dtype", "3"},
                                                                         {"indices", "3"},
                                                                         {"IDS_version", "3"}});

    int rc = east_plugin(&interface);

    delete interface.request_block->nameValueList.nameValue;
    delete interface.request_block;
    delete interface.data_block;
    delete interface.userdefinedtypelist;
    delete interface.logmalloclist;

    REQUIRE(rc == 0);
    // const uda::Result &result = client.get("EAST::help(element='equilibrium/ids_properties/homogeneous_time', indices='', experiment='EAST', dtype=3, shot=" SHOT_NUM ", IDS_version='3')", "");

    // uda::Data *data = result.data();
    // REQUIRE(data != nullptr);
    // REQUIRE(!data->isNull());
    // REQUIRE(data->type().name() == typeid(int).name());

    // auto val = dynamic_cast<uda::Scalar *>(data);

    // REQUIRE(val != nullptr);
    // REQUIRE(!val->isNull());

    // REQUIRE(val->type().name() == typeid(int).name());
    // REQUIRE(val->as<int>() == 0);
}
// TEST_CASE("Test time_slice/Shape_of", "[IMAS][EAST]")
// {
// #ifdef FATCLIENT
// #include "setup.inc"
// #endif

//     uda::Client client;

//     const uda::Result &result = client.get("EAST::read(element='equilibrium/time_slice/Shape_of', indices='', experiment='EAST', dtype=3, shot=" SHOT_NUM ", IDS_version='3')", "");

//     REQUIRE(result.errorCode() == 0);
//     REQUIRE(result.errorMessage().empty());

//     uda::Data *data = result.data();

//     REQUIRE(data != nullptr);
//     REQUIRE(!data->isNull());
//     REQUIRE(data->type().name() == typeid(int).name());

//     auto val = dynamic_cast<uda::Scalar *>(data);

//     REQUIRE(val != nullptr);
//     REQUIRE(!val->isNull());

//     REQUIRE(val->as<int>() == 2);
//     REQUIRE(val->type().name() == typeid(int).name());
// }

// /*
//  * Test equilibrium profiles_2d/psi
//  */
// TEST_CASE("Test Test equilibrium/time_slice/#/profiles_2d/#/psi", "[IMAS][EAST]")
// {
// #ifdef FATCLIENT
// #include "setup.inc"
// #endif

//     uda::Client client;

//     const uda::Result &result = client.get("EAST::read(element='equilibrium/time_slice/#/profiles_2d/#/psi', indices='1;1', experiment='EAST', dtype=7, shot=" SHOT_NUM ", IDS_version='3')", "");
//     const uda::Result &result2 = client.get("EAST::read(element='equilibrium/time_slice/#/profiles_2d/#/psi', indices='1;2', experiment='EAST', dtype=7, shot=" SHOT_NUM ", IDS_version='3')", "");

//     REQUIRE(result.errorCode() == 0);
//     REQUIRE(result.errorMessage().empty());

//     uda::Data *data = result.data();

//     REQUIRE(data != nullptr);
//     REQUIRE(!data->isNull());
//     REQUIRE(data->type().name() == typeid(double).name());

//     auto arr = dynamic_cast<uda::Array *>(data);

//     // std::cout<< "rank  =" << arr->shape().size()<<std::endl;
//     // std::cout<< "shape =" << arr->shape()[0]<< "," <<arr->shape()[1] <<std::endl;

//     // auto shape=arr->shape();
//     // auto vals = arr->as<double>();
//     // int width=shape[0],height=shape[1];
//     // for (int i=0;i<height;++i)
//     // {
//     //     for (int j=0;j<width;++j)
//     //     {
//     //         std::cout<< vals[i*width+j]<<' ';
//     //     }
//     //     std::cout<<std::endl;
//     // }

//     REQUIRE(arr != nullptr);
//     REQUIRE(!arr->isNull());

//     std::vector<double> expected = {0.6033437862, 0.6037969238, 0.6042322372, 0.6046487414, 0.6050455565};

//     REQUIRE(arr->type().name() == typeid(double).name());

//     auto vals = arr->as<double>();
//     vals.resize(5);

//     REQUIRE(vals == ApproxVector(expected));

//     auto *data2 = result2.data();
//     auto arr2 = dynamic_cast<uda::Array *>(data2);
//     auto vals2 = arr2->as<double>();

//     REQUIRE(vals2 == ApproxVector(expected));
// }
