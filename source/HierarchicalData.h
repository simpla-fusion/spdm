
#ifndef SP_HIERACHICAL_DATA_H_
#define SP_HIERACHICAL_DATA_H_
#include "utility/Logger.h"
#include <any>
#include <array>
#include <complex>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace sp
{
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

template <typename T,
          typename TObject = std::map<std::string, T>,
          typename TArray = std::vector<T>,
          typename TBlock = std::tuple<std::shared_ptr<void>, DataType, std::vector<size_t>>>
using DataUnion = std::variant<
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

/**
 * Hierarchical Data Struct
*/
class HierarchicalData : public DataUnion<HierarchicalData>
{
public:
    typedef DataUnion<HierarchicalData> base_type;
    typedef HierarchicalData this_type;

    HierarchicalData() = default;
    template <typename V>
    HierarchicalData(const V& v) : base_type(v) {}
    HierarchicalData(const HierarchicalData& other) : base_type(other) {}
    HierarchicalData(HierarchicalData&& other) : base_type(std::move(other)) {}
    ~HierarchicalData() = default;

    void swap(this_type& other) { base_type::swap(other); }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    };
    void clear()
    {
        if (index() == DataType::Array)
        {
            std::get<DataType::Array>(*this).clear();
        }
        else if (index() == DataType::Object)
        {
            std::get<DataType::Object>(*this).clear();
        }
    }

    size_t size() const
    {
        if (index() <= DataType::Array)
        {
            return std::get<DataType::Array>(*this).size();
        }
        else if (index() <= DataType::Object)
        {
            return std::get<DataType::Object>(*this).size();
        }
        else
        {
            return 0;
        }
    }

    bool has_a(const std::string& key) const
    {
        return index() == DataType::Object && std::get<DataType::Object>(*this).find(key) != std::get<DataType::Object>(*this).end();
    }
    template <typename V>
    bool operator==(const V& v) const
    {
        return *this == this_type(v);
    }

    template <typename V>
    this_type& operator=(const V& v)
    {
        emplace<V>(v);
        return *this;
    }

    template <typename V>
    auto as() const { return std::get<V>(*this); }

    template <DataType V>
    auto as() const { return std::get<V>(*this); }

    const this_type& at(const std::string& key) const
    {
        if (index() != DataType::Object)
        {
            throw std::out_of_range(key);
        }

        return std::get<DataType::Object>(*this).at(key);
    }

    this_type& get(const std::string& key)
    {
        if (index() == DataType::Null)
        {
            base_type::emplace<DataType::Object>();
        }
        else if (index() != DataType::Object)
        {
            throw std::out_of_range(key);
        }

        return std::get<DataType::Object>(*this)[key];
    }

    void remove(const std::string& key)
    {
        if (index() == DataType::Object)
        {
            std::get<DataType::Object>(*this).erase(key);
        }
    }
    void resize(size_t s)
    {
        if (index() == DataType::Array)
        {
            std::get<DataType::Array>(*this).resize(s);
        }
    }
    template <typename V>
    this_type& push_back()
    {
        if (index() == DataType::Null)
        {
            base_type::emplace<DataType::Array>();
        }
        else if (index() != DataType::Object)
        {
            throw std::runtime_error("illegal type");
        }
        std::get<DataType::Array>(*this).push_back(this_type());
        return std::get<DataType::Array>(*this).back();
    }
    void pop_back()
    {
        if (index() == DataType::Array)
        {
            std::get<DataType::Array>(*this).pop_back();
        }
        else if (index() == DataType::Null)
        {
            throw std::runtime_error("illegal type");
        }
    }

    this_type& item(int idx)
    {
        if (index() != DataType::Array)
        {
            throw std::runtime_error("illegal type !");
        }
        auto& v = std::get<DataType::Array>(*this);
        size_t size = v.size();

        if (size == 0)
        {
            throw std::out_of_range("");
        }

        return v[(idx + size) % size];
    }
    const this_type& item(int idx) const
    {
        if (index() != DataType::Array)
        {
            throw std::runtime_error("illegal type !");
        }
        const auto& v = std::get<DataType::Array>(*this);
        size_t size = v.size();

        if (size == 0)
        {
            throw std::out_of_range(FILE_LINE_STAMP_STRING);
        }

        return v[(idx + size) % size];
    }

    this_type& operator[](const std::string& key) { return get(key); }
    const this_type& operator[](const std::string& key) const { return at(key); }

    this_type& operator[](int idx) { return item(idx); }
    const this_type& operator[](int idx) const { return item(idx); }
};

} // namespace sp

#endif //SP_HIERACHICAL_DATA_H_