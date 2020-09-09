#ifndef SPDB_DATABLOCK_
#define SPDB_DATABLOCK_

#include <memory>
#include <vector>
// #ifdef __cplusplus
// extern "C"
// {
// #endif

// #define SpObject_INTERFACE_HEAD ;

//     typedef struct
//     {
//         SpObject_INTERFACE_HEAD;

//     } SpObjectInterface;

typedef struct
{
    // SpObject_INTERFACE_HEAD;

    char* data;
    unsigned int element_size;
    int dtype;
    int nd;
    unsigned int* dimensions;
    unsigned int* strides;
    int flags;
    char _[];
} DataBlock;

// #ifdef __cplusplus
// }
// #endif

namespace sp::db
{
class DataBlock
{
    std::shared_ptr<void> m_data_;
    std::vector<size_t> m_dimensions_;

public:
    DataBlock() = default;
    ~DataBlock() = default;

    DataBlock(int nd, const size_t* dimensions, void* data = nullptr, int element_size = sizeof(double));

    template <typename TDIM>
    DataBlock(int nd, const TDIM* dimensions);

    DataBlock(DataBlock const&);
    DataBlock(DataBlock&&);

    void swap(DataBlock& other);

    DataBlock& operator=(const DataBlock& other)
    {
        DataBlock(other).swap(*this);
        return *this;
    }

    std::type_info const& value_type_info() const;
    bool is_slow_first() const;

    template <typename TDim>
    void reshape(int ndims, const TDim* dims)
    {
        std::vector<size_t>(dims, dims + ndims).swap(m_dimensions_);
    }

    const std::vector<size_t>& shape()const;

    void* data();

    const void* data() const;

    size_t element_size() const;
    size_t ndims() const;
    size_t const* dims() const;

    template <typename U>
    U* as();
    template <typename U>
    const U* as() const;

    DataBlock slice(const std::tuple<int, int, int>& slice)
    {
        return DataBlock{};
    }
    DataBlock slice(const std::tuple<int, int, int>& slice) const
    {
        return DataBlock{};
    }
};
} // namespace sp::db
#endif // SPDB_DATABLOCK_