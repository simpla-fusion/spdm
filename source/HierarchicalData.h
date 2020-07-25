
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
    Object,
    Array,
    Null,
    Block,
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
    Other
};

template <typename T>
using ObjectContainer = std::map<std::string, T>;
template <typename T>
using ArrayContainer = std::vector<T>;

typedef std::tuple<std::shared_ptr<void>, DataType, std::vector<size_t>> block_type;
/**
 * Hierarchical Data Struct
*/
template <template <typename> class TmplObject = ObjectContainer,
          template <typename> class TmplArray = ArrayContainer,
          typename... ElementTypes>
class HierarchicalDataTmpl : public std::variant<
                                 TmplObject<HierarchicalDataTmpl<TmplObject, TmplArray, ElementTypes...>>,
                                 TmplArray<HierarchicalDataTmpl<TmplObject, TmplArray, ElementTypes...>>,
                                 ElementTypes...>
{
public:
    typedef HierarchicalDataTmpl<TmplObject, TmplArray, ElementTypes...> this_type;

    typedef std::variant<
        TmplObject<this_type>,
        TmplArray<this_type>,
        ElementTypes...>
        base_type;

    HierarchicalDataTmpl() : base_type(nullptr) {}

    template <typename V>
    HierarchicalDataTmpl(const V& v) : base_type(v) {}

    HierarchicalDataTmpl(const this_type& other) : base_type(other) {}

    HierarchicalDataTmpl(this_type&& other) : base_type(std::move(other)) {}

    ~HierarchicalDataTmpl() = default;

    void swap(this_type& other) { base_type::swap(other); }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void clear()
    {
        if (base_type::index() == DataType::Array)
        {
            std::get<DataType::Array>(*this).clear();
        }
        else if (base_type::index() == DataType::Object)
        {
            std::get<DataType::Object>(*this).clear();
        }
    }

    size_t size() const
    {
        if (base_type::index() == DataType::Array)
        {
            return std::get<DataType::Array>(*this).size();
        }
        else if (base_type::index() == DataType::Object)
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
        return base_type::index() == DataType::Object &&
               std::get<DataType::Object>(*this).find(key) != std::get<DataType::Object>(*this).end();
    }
    
    template <typename V>
    bool operator==(const V& v) const { return *this == this_type(v); }

    template <typename V>
    this_type& operator=(const V& v)
    {
        this->template emplace<V>(v);
        return *this;
    }

    template <typename V>
    auto as() const { return std::get<V>(*this); }

    template <DataType V>
    auto as() const { return std::get<V>(*this); }

    auto& as_array()
    {
        if (base_type::index() == DataType::Null)
        {
            base_type::template emplace<DataType::Array>();
        }
        if (base_type::index() != DataType::Array)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }
        return std::get<DataType::Array>(*this);
    }

    const auto& as_array() const
    {
        if (base_type::index() != DataType::Array)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }
        return std::get<DataType::Array>(*this);
    }

    auto& as_object()
    {
        if (base_type::index() == DataType::Null)
        {
            base_type::template emplace<DataType::Object>();
        }
        if (base_type::index() != DataType::Object)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }

        return std::get<DataType::Object>(*this);
    }

    const auto& as_object() const
    {
        if (base_type::index() != DataType::Object)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }

        return std::get<DataType::Object>(*this);
    }

    const this_type& at(const std::string& key) const { return as_object().at(key); }

    this_type& get(const std::string& key) { return as_object()[key]; }

    void remove(const std::string& key)
    {
        if (base_type::index() == DataType::Object)
        {
            as_object().erase(key);
        }
    }

    void resize(size_t s) { as_array().resize(s); }

    template <typename V>
    this_type& push_back()
    {
        auto& arr = as_array();
        arr.push_back(this_type());
        return arr.back();
    }

    void pop_back()
    {
        try
        {
            as_array().pop_back();
        }
        catch (...)
        {
        }
    }

    this_type& item(int idx)
    {
        auto& arr = as_array();
        size_t size = arr.size();

        if (size == 0)
        {
            throw std::out_of_range("");
        }

        return arr[(idx + size) % size];
    }

    const this_type& item(int idx) const
    {
        const auto& arr = as_array();
        size_t size = arr.size();

        if (size == 0)
        {
            throw std::out_of_range(FILE_LINE_STAMP_STRING);
        }

        return arr[(idx + size) % size];
    }

    this_type& operator[](const std::string& key) { return get(key); }

    const this_type& operator[](const std::string& key) const { return at(key); }

    this_type& operator[](int idx) { return item(idx); }

    const this_type& operator[](int idx) const { return item(idx); }
};

typedef HierarchicalDataTmpl<
    ObjectContainer,       //Object
    ArrayContainer,        //Array
    std::nullptr_t,        //Null
    block_type,            //Block
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
    std::any               //Other
    >
    HierarchicalData;
} // namespace sp

#endif //SP_HIERACHICAL_DATA_H_