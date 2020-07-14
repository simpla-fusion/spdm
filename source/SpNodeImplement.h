#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include "SpNode.h"
#include "SpUtil.h"
#include <algorithm>
#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>
namespace sp
{

    template <typename TAG = nullptr_t>
    class SpEntryTmpl;
    typedef SpEntryTmpl<> node_t;

    template <typename TAG = nullptr_t>
    class AttributesTmpl;
    typedef AttributesTmpl<> attributes_t;

    template <typename TAG = nullptr_t, NodeTag tag = NodeTag::Null>
    class ContentTmpl;
    typedef ContentTmpl<> content_t;

    template <typename TAG>
    class AttributesTmpl : public Attributes
    {
        AttributesTmpl() = default;
        virtual ~AttributesTmpl() = default;
        AttributesTmpl(AttributesTmpl const &) = default;
        AttributesTmpl(AttributesTmpl &&) = default;

        AttributesTmpl &operator=(AttributesTmpl const &other)
        {
            AttributesTmpl(other).swap(*this);
            return *this;
        };
        void swap(AttributesTmpl &) {}
        AttributesTmpl *copy() const { return new AttributesTmpl(*this); }

        std::any get(std::string const &name) const;         // get attribute, return nullptr is name does not exist
        int set(std::string const &name, std::any const &v); // set attribute
        int remove(std::string const &name);                 // remove attribuet
    };

    template <typename TAG>
    class ContentTmpl<TAG> : public Content
    {
    };

    template <typename TAG>
    class ContentTmpl<TAG, NodeTag::Scalar> : public ContentTmpl<TAG>
    {
    public:
        void swap(ContentTmpl<TAG, NodeTag::Scalar> &other);
        std::any value();
        void value(std::any const &v);
    };

    template <typename TAG>
    class ContentTmpl<TAG, NodeTag::Block> : public ContentTmpl<TAG>
    {
    public:
        void swap(ContentTmpl<TAG, NodeTag::Block> &other);

        void data(std::shared_ptr<char> const &);
        std::shared_ptr<char> data();
        std::shared_ptr<char> data() const;

        void dims(std::vector<size_t> const &);
        const std::vector<size_t> &dims() const;

        void element_size(size_t s);
        size_t element_size() const;

        template <typename V>
        void data(std::shared_ptr<V> const &v, int nd, size_t const *d)
        {
            data(std::reinterpret_pointer_cast<char>(v));
            element_size(sizeof(V));
            std::vector<size_t> t_dims(d, d + nd);
            this->dims(t_dims);
        }

        template <typename V>
        std::shared_ptr<V> data() { return std::reinterpret_pointer_cast<char>(data()); }
    };

    template <typename TAG>
    class ContentTmpl<TAG, NodeTag::List> : public ContentTmpl<TAG>
    {
    public:
        void swap(ContentTmpl<TAG, NodeTag::List> &other);

        size_t size() const;
        Range<Iterator<SpEntryTmpl<TAG>>, Iterator<SpEntryTmpl<TAG>>> children(); // return children
        std::shared_ptr<node_t> child(int);                                       // return node at idx,  if idx >size() then throw exception
        std::shared_ptr<node_t> append();                                         // insert new child node after last child
        Iterator<SpEntryTmpl<TAG>> first_child();                                 // return first child node, next(first_child()) return the nearest slibing
        int remove_child(int idx);                                                // remove child with key, return true if key exists
    };

    template <typename TAG>
    class ContentTmpl<TAG, NodeTag::Object> : public ContentTmpl<TAG>
    {
    public:
        void swap(ContentTmpl<TAG, NodeTag::Object> &other);
        size_t size() const;
        Range<Iterator<SpEntryTmpl<TAG>>, Iterator<SpEntryTmpl<TAG>>> children(); // return children
        std::shared_ptr<SpEntryTmpl<TAG>> find(std::string const &);              // return node at key,  if key does not exist then return nullptr
        std::shared_ptr<SpEntryTmpl<TAG>> child(std::string const &);             // return node at key,  if key does not exist then create one
        int remove_child(std::string const &key);                                 // remove child with key, return true if key exists
    };
    //----------------------------------------------------------------------------------------------------------

    template <typename TAG>
    std::ostream &operator<<(std::ostream &os, SpEntryTmpl<TAG> const &d) { return d.repr(os); }

} // namespace sp

#endif // SP_ENTRY_H_