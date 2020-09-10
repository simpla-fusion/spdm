#ifndef SP_DATABLOCK_H_
#define SP_DATABLOCK_H_

#include <memory>
#include <vector>

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

    std::type_info const& value_type_info() const { return typeid(double); }
    bool is_slow_first() const { return true; }

    template <typename TDim>
    void reshape(int ndims, const TDim* dims)
    {
        std::vector<size_t>(dims, dims + ndims).swap(m_dimensions_);
    }

    const std::vector<size_t>& shape() const { return m_dimensions_; }

    void* data() { return m_data_.get(); }

    const void* data() const { return m_data_.get(); }

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
#endif // SP_DATABLOCK_H_