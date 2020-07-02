//
// Created by salmon on 18-3-15.
//

#ifndef SIMPLA_SPDMBLOCK_H
#define SIMPLA_SPDMBLOCK_H

#include "SpDM.h"
namespace simpla {
namespace data {

struct SpDMBlock : public spObject {
    SP_OBJECT_HEAD(spObject, SpDMBlock)

   public:
    typedef std::ptrdiff_t difference_type;
    typedef std::size_t size_type;
    SpDMBlock() = default;
    ~SpDMBlock() override = default;
    SpDMBlock(SpDMBlock const &other) = default;
    SpDMBlock(SpDMBlock &&other) noexcept : base_type(std::move(other)) {}

    void swap(this_type &other) { base_type::swap(other); }
    template <typename B>
    void reset(B const &b){};
    void Clear(){};
    void FillNaN(){};

    size_type size() const override { return 0; }
    this_type *Clone() const override { return new this_type(); }
    this_type *Copy() const override { return new this_type(*this); }
    void DeepCopy(base_type const &other) override { this_type(*this).swap(*this); }
    bool Equal(base_type const &other) const override { return false; }
    bool Less(base_type const &other) const override { return false; }
    int Accept(visitor_type const &visitor) const override {
        int res = base_type::Accept(visitor);
        //        if (auto p = dynamic_cast<visitor_type *>(&visitor)) { res = p->Block(this); }
        return res;
    }
    unsigned int value_type_tag() const { return kNull; };

    size_type ndim() const { return 0; }
    size_type dimensions(std::size_t *) const { return 0; }

    virtual size_type item_size_in_byte() const { return 0; }
    virtual size_type size_in_byte() const { return size() * item_size_in_byte(); }
    virtual void stride_in_byte(std::size_t *) const {}

    virtual void shape(size_type *) const {}
    virtual void start(difference_type *) const {}
    virtual void count(size_type *) const {}
    virtual void step(size_type *) const {}

    virtual int reshape(std::size_t *start, std::size_t *count, std::size_t *strides, unsigned int ndim,
                        unsigned int tag) {
        return 0;
    }

    int CopyIn(this_type const &other) { return kSuccessful; }
    value_type *pointer() { return nullptr; }
    value_type const *pointer() const { return nullptr; }
    this_type DuplicateArray() const { return this_type(); }
};
}  // namespace data
typedef data::SpDMBlock spBlock;
}  // namespace simpla
#endif  // SIMPLA_SPDMBLOCK_H
