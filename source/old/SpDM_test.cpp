//
// Created by salmon on 18-1-19.
//

#include <gtest/gtest.h>
#include <spdm/SpDM.h>

#include "SpDM_test.h"

typedef testing::Types<simpla::SpDMElement<char>> TypeParamList;

INSTANTIATE_TYPED_TEST_CASE_P(DataModel, TestDataModel, TypeParamList);