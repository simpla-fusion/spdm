#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_

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

        // attributes
        const Attributes &attributes() const { return *m_attributes_; } // return list of attributes
        Attributes &attributes() { return *m_attributes_; }             // return list of attributes

        //----------------------------------------------------------------------------------------------------------
        // content
        Content &content() { return *m_content_; }
        const Content &content() const { return *m_content_; }

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        template <typename T>
        std::shared_ptr<Node> operator[](T const &v) { return child(v); }

        //----------------------------------------------------------------------------------------------------------
        // level 0
        std::shared_ptr<Node> parent() const; // return parent node
        void remove();                        // remove self from parent

        //----------------------------------------------------------------------------------------------------------
        // level 1
        // range select(XPath const &path) const;                       // select from children
        // std::shared_ptr<Node> select_one(XPath const &path) const; // return the first selected child

    protected:
        Entry(Attributes *attr, Content *content);

    private:
        std::unique_ptr<Content> m_content_;
        std::unique_ptr<Attributes> m_attributes_;
    };

} // namespace sp
#endif //SP_ENTRY_H_