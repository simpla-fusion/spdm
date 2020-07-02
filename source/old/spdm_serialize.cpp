//
// Created by salmon on 18-2-27.
//

#include <gtest/gtest.h>
#include "spdm/SpDM.h"
#include "spdm/SpDMIOStream.h"
#include "spdm/SpDMJSON.h"
#include "spdm/Serializable.h"
using namespace simpla::data;
struct gTorus : public Serializable<SpDM> {
    gTorus() = default;
    ~gTorus() = default;
    SP_PROPERTY(double, MajorRadius) = 0;
    SP_PROPERTY(double, MinorRadius) = 0;
    SP_PROPERTY(double, MaxMajorAngle) = 0;
    SP_PROPERTY(double, MinMajorAngle) = 0;
    SP_PROPERTY(double, MinMinorAngle) = 0;
    SP_PROPERTY(double, MaxMinorAngle) = 0;
};

int main(int argc, char** argv) {
    gTorus torus;
    SpDM db;
    SpDM context;
    torus.Context(context);
    torus.Serialize(db);

    std::cout << "@context:" << context << std::endl;

    std::cout << db << std::endl;

    std::cout << "[JSON]" << std::endl;

    WriteJSON(db, std::cout);

    std::cout << std::endl << "The End!" << std::endl;
}