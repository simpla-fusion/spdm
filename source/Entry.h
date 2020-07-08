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

    //##############################################################################################################
    class Attributes
    {
    public:
        typedef Attributes this_type;

        static Attributes *create();
        Attributes *copy() const;

        Attributes() = default;
        Attributes(this_type const &) = default;
        Attributes(this_type &&) = default;
        virtual ~Attributes() = default;

        Attributes &operator=(this_type const &other);

        void swap(this_type &);

        std::any get(std::string const &name) const;         // get attribute, return nullptr is name does not exist
        int set(std::string const &name, std::any const &v); // set attribute
        int remove(std::string const &name);                 // remove attribuet
    };

    //##############################################################################################################
    class Content
    {
    public:
        typedef Content this_type;

        static Content *create(int tag);
        Content *as(int tag);
        Content *copy() const;

        Content() = default;
        virtual ~Content() = default;
        Content(this_type const &) = default;
        Content(this_type &&) = default;
        Content &operator=(this_type const &other);
        void swap(Content &);

        int type() const;
    };

    //##############################################################################################################
    class Entry
    {
    public:
        typedef Entry this_type;

        static Entry *create(int tag = 0);

        Entry *copy() const;

        Entry(int tag = 0);

        Entry(this_type const &other);

        Entry(this_type &&other);

        virtual ~Entry();

        this_type &operator=(this_type const &other);

        void swap(this_type &other);

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

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node

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

    protected:
        Entry(Attributes *attr, Content *content);

    private:
        std::unique_ptr<Content> m_content_;
        
        std::unique_ptr<Attributes> m_attributes_;
    };

} // namespace sp
#endif //SP_ENTRY_H_