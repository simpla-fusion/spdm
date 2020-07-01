#include "SpDB.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("SpDB Create", "[SpDB]")
{
    setenv("SPDB_CONFIG_PATH", "/workspaces/SpDB/", 1);

    SpDB db;

    db.connect("EAST","imas/3");

    auto doc = db.open(1234);
  
    doc.fetch("/equilibrium/time_slice/profiles_2d[@id='1']/psi#itime=3");
}