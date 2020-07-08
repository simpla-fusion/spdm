#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include "Range.h"
#include "Util.h"
#include "XPath.h"
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
    class Node;

    class Entry
    {
    public:
        typedef Entry this_type;

        Entry() = default;
        Entry(this_type const &other) = default;
        Entry(this_type &&other) = default;
        virtual ~Entry() = default;
        this_type &operator=(this_type const &other) = default;

        void swap(this_type &other);

        static Entry *create(std::string const &backend = "");

        Entry *copy() const;

        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        int type() const;

        //----------------------------------------------------------------------------------------------------------
        // as leaf node,  need node.type = Scalar || Block
        //----------------------------------------------------------------------------------------------------------
        typedef std::tuple<std::shared_ptr<char> /*data pointer*/, int /*element size*/, std::vector<size_t> /*dimensions*/> block_type;

        std::any as_scalar() const; // get value , if value is invalid then throw exception

        void as_scalar(std::any); // set value , if fail then throw exception

        block_type as_block() const; // get block

        void as_block(block_type const &); // set block

        Entry &as_list();

        Entry &as_object();

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        // function level 0
        std::shared_ptr<Node> child(std::string const &); // return reference of child node , if key does not exists then insert new

        std::shared_ptr<const Node> child(std::string const &) const; // return reference of child node , if key does not exists then insert new

        std::shared_ptr<Node> child(int idx); // return reference i-th child node , if idx does not exists then throw exception

        std::shared_ptr<const Node> child(int idx) const; // return reference i-th child node , if idx does not exists then throw exception

        std::shared_ptr<Node> append(); // append node to tail of list , return reference of new node

        void remove_child(int idx); // remove i-th child

        void remove_child(std::string const &key); // remove child at key

        void remove_children(); // remove children , set node.type => Null

        Range<Iterator<std::shared_ptr<Node>>> children() const; // reutrn list of children

        // level 1
        Range<Iterator<std::shared_ptr<Node>>> select(XPath const &path) const; // select from children

        std::shared_ptr<Node> select_one(XPath const &path) const; // return the first selected child
    };

} // namespace sp
#endif //SP_ENTRY_H_