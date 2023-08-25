#include "utility/Iterator.h"
#include "utility/Range.h"
#include <iostream>
#include <memory>
#include <vector>
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

TEST_CASE("Iterator", "[SpDB]")
{
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0, 5.0};

    sp::Iterator<double> ib(v.begin());
    sp::Iterator<double> ie(v.end());

    for (; ib != ie; ++ib)
    {
        std::cout << *ib << " ";
    }
    std::cout << std::endl;

    double w[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    sp::Range<double> r(&w[0], w + 4);

    for (const auto& v : r)
    {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    std::shared_ptr<double> p(new double[5]);
    sp::Range<double> r2(p, 4);

    double count = 0;

    for (auto& v : r2)
    {
        ++count;
        v = 1.1 + count;
    }
    
    for (auto& v : r2)
    {
        std::cout << v << " ";
    }
    
    std::cout << std::endl;
}