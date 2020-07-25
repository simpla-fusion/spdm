#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_

enum DataType
{
    Null = 0,
    Object = 1,
    Array = 2,
    Block = 3,
    String,
    Boolean,
    Integer,
    Long,
    Float,
    Double,
    Complex,
    IntVec3,
    LongVec3,
    FloatVec3,
    DoubleVec3,
    Custom
};

#ifdef __cplusplus
#include <any>
#include <array>
#include <complex>
#include <map>
#include <memory>
#include <tuple>
#include <variant>
#include <vector>

namespace sp
{
template <typename T,
          typename TObject = std::map<std::string, T>,
          typename TArray = std::vector<T>,
          typename TBlock = std::tuple<std::shared_ptr<void>, DataType, std::vector<size_t>>>
using DataUnionType = std::variant<
    std::nullptr_t,
    TObject,               //Object = 1,
    TArray,                //Array = 2,
    TBlock,                //Block = 3,
    std::string,           //String,
    bool,                  //Boolean,
    int,                   //Integer,
    long,                  //Long,
    float,                 //Float,
    double,                //Double,
    std::complex<double>,  //Complex,
    std::array<int, 3>,    //IntVec3,
    std::array<long, 3>,   //LongVec3,
    std::array<float, 3>,  //FloatVec3,
    std::array<double, 3>, //DoubleVec3,
    std::any>;
}
#endif
#endif // DATA_TYPE_H_