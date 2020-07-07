#ifndef SPDB_IMPLEMENT_H_
#define SPDB_IMPLEMENT_H_
#include "SpNode.h"
#include <any>
#include <map>
#include <memory>
namespace sp
{
    template <typename... Args>
    class SpEntryT;

    class SpEntry : public std::enable_shared_from_this<SpEntry>
    {
    public:
        friend class SpNode;
        typedef std::shared_ptr<SpEntry> node_type;
        typedef std::pair<node_type, node_type> range_type;

        SpEntry(){};
        virtual ~SpEntry(){};

        template <typename... TArgs>
        static std::shared_ptr<SpEntry> create(std::shared_ptr<SpNode> const &parent)
        {
            return std::make_shared<SpEntryT<TArgs...>>(parent);
        }
        virtual size_t size() const = 0;
        virtual size_t depth() const = 0;                                      // distance(root())
        virtual void remove() = 0;                                             // remove self
        virtual node_type parent() const = 0;                                  // return parent node
        virtual SpNode::TypeOfNode type() const = 0;                           //
        virtual node_type copy() const = 0;                                    //
        virtual node_type create(SpNode::TypeOfNode) const = 0;                //
        virtual std::map<std::string, std::any> attributes() const = 0;        // return list of attributes
        virtual std::any attribute(std::string const &name) const = 0;         // get attribute, return nullptr is name does not exist
        virtual int attribute(std::string const &name, std::any const &v) = 0; // set attribute
        virtual int remove_attribute(std::string const &name) = 0;             // remove attribute
        virtual void value(std::any const &) = 0;                              // set value
        virtual std::any value() const = 0;                                    // get value
        virtual range_type children() const = 0;                               // return children
        virtual node_type first_child() const = 0;                             // return first child node
        virtual node_type child(int) = 0;                                      // return node at idx,  if idx >size() then throw exception
        virtual node_type child(int) const = 0;                                // return node at idx,  if idx >size() then throw exception
        virtual node_type insert_before(int pos) = 0;                          // insert new child node before pos, return new node
        virtual node_type insert_after(int pos) = 0;                           // insert new child node after pos, return new node
        virtual node_type prepend() = 0;                                       // insert new child node before first child
        virtual node_type append() = 0;                                        // insert new child node afater last child
        virtual int remove_child(int idx) = 0;                                 // remove child at pos, return true if idx exists
        virtual node_type child(std::string const &) const = 0;                // return node at key,  if key does not exist then throw exception
        virtual node_type child(std::string const &) = 0;                      // return node at key,  if key does not exist then create one
        virtual int remove_child(std::string const &key) = 0;                  // remove child with key, return true if key exists
        virtual range_type select(SpXPath const &path) const = 0;              // select from children
        virtual node_type select_one(SpXPath const &path) const = 0;           // return the first selected child
    };

    template <typename... Args>
    class SpEntryT : public SpEntry
    {
    public:
        SpEntryT();
        SpEntryT(SpEntryT const &);
        SpEntryT(SpEntryT &&);
        ~SpEntryT();
        SpEntryT(std::shared_ptr<SpNode> const &parent);

        size_t size() const;
        size_t depth() const;                                      // distance(root())
        void remove();                                             // remove self
        node_type parent() const;                                  // return parent node
        SpNode::TypeOfNode type() const;                           //
        node_type copy() const;                                    //
        node_type create(SpNode::TypeOfNode) const;                //
        std::map<std::string, std::any> attributes() const;        // return list of attributes
        std::any attribute(std::string const &name) const;         // get attribute, return nullptr is name does not exist
        int attribute(std::string const &name, std::any const &v); // set attribute
        int remove_attribute(std::string const &name);             // remove attribute
        void value(std::any const &);                              // set value
        std::any value() const;                                    // get value
        range_type children() const;                               // return children
        node_type first_child() const;                             // return first child node
        node_type child(int);                                      // return node at idx,  if idx >size() then throw exception
        node_type child(int) const;                                // return node at idx,  if idx >size() then throw exception
        node_type insert_before(int pos);                          // insert new child node before pos, return new node
        node_type insert_after(int pos);                           // insert new child node after pos, return new node
        node_type prepend();                                       // insert new child node before first child
        node_type append();                                        // insert new child node afater last child
        int remove_child(int idx);                                 // remove child at pos, return true if idx exists
        node_type child(std::string const &) const;                // return node at key,  if key does not exist then throw exception
        node_type child(std::string const &);                      // return node at key,  if key does not exist then create one
        int remove_child(std::string const &key);                  // remove child with key, return true if key exists
        range_type select(SpXPath const &path) const;              // select from children
        node_type select_one(SpXPath const &path) const;           // return the first selected child
    private:
        struct pimpl_s;
        std::unique_ptr<pimpl_s> m_pimpl_;
    };

    struct entry_tag_in_memory;
    struct entry_tag_jason;

} // namespace sp
#endif //SPDB_IMPLEMENT_H_