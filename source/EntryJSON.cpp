#include "Entry.h"
#include "Node.h"
#include <any>
#include <map>
#include <vector>

namespace sp
{
    class EntryXML : public Entry
    {
    public:
        typedef EntryXML this_type;

        EntryXML();

        EntryXML(EntryXML const &other);

        EntryXML(EntryXML &&other);

        ~EntryXML();

        EntryXML &operator=(EntryXML const &other);

        void swap(EntryXML &other);

        Entry *copy() const;

        Entry *move();

        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        int type() const;

        //----------------------------------------------------------------------------------------------------------
        // attribute
        //----------------------------------------------------------------------------------------------------------
        bool has_attribute(std::string const &k) const; // if key exists then return true else return false

        bool check_attribute(std::string const &k, std::any const &v) const; // if key exists and value ==v then return true else return false

        std::any get_attribute(std::string const &key) const; // get attribute at key, if key does not exist return nullptr

        std::any get_attribute(std::string const &key, std::any const &default_value = std::any{}); // get attribute at key, if key does not exist return nullptr

        void set_attribute(std::string const &key, std::any const &v); // set attribute at key as v

        void remove_attribute(std::string const &key = ""); // remove attribute at key, if key=="" then remove all

        Range<Iterator<const std::pair<const std::string, std::any>>> attributes() const; // return reference of  all attributes

        //----------------------------------------------------------------------------------------------------------
        // as leaf node,  need node.type = Scalar || Block
        //----------------------------------------------------------------------------------------------------------

        std::any get_scalar() const; // get value , if value is invalid then throw exception

        void set_scalar(std::any const &); // set value , if fail then throw exception

        std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>> get_raw_block() const; // get block

        void set_raw_block(std::shared_ptr<char> const &, std::type_info const &, std::vector<size_t> const &); // set block

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        // function level 0
        std::shared_ptr<const Node> find_child(std::string const &) const; // return reference of child node , if key does not exists then insert new

        std::shared_ptr<Node> find_child(std::string const &); // return reference of child node , if key does not exists then insert new

        std::shared_ptr<Node> append(); // append node to tail of list , return reference of new node

        std::shared_ptr<Node> append(std::shared_ptr<Node> const &);

        void append(const Iterator<std::shared_ptr<Node>> &b, const Iterator<std::shared_ptr<Node>> &); // insert node to object

        std::shared_ptr<Node> insert(std::string const &key, std::shared_ptr<Node> const &n = nullptr); // insert node to object

        void insert(Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &b,
                    Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &e); // insert node to object

        std::shared_ptr<Node> child(std::string const &key); // get child, create new if key does not  exist

        // std::shared_ptr<const Node> child(std::string const &key) const; // get child, create new if key does not  exist

        std::shared_ptr<Node> child(int idx); // return reference i-th child node , if idx does not exists then throw exception

        // std::shared_ptr<const Node> child(int idx) const; // return reference i-th child node , if idx does not exists then throw exception

        void remove_child(std::string const &key); // remove child at key

        void remove_child(int idx); // remove i-th child

        void remove_children(); // remove children , set node.type => Null

        // std::pair<Iterator<const Node>, Iterator<const Node>> children() const; // reutrn list of children

        std::pair<Iterator<Node>, Iterator<Node>> children(); // reutrn list of children

        // level 1
        // std::pair<Iterator<const Node>, Iterator<const Node>> select(XPath const &path) const; // select from children

        std::pair<Iterator<Node>, Iterator<Node>> select(XPath const &path); // select from children

        // std::shared_ptr<const Node> select_one(XPath const &path) const; // return the first selected child

        std::shared_ptr<Node> select_one(XPath const &path); // return the first selected child

    private:
        std::unique_ptr<Attributes> m_attributes_;

        std::unique_ptr<Content> m_content_;

        std::vector<std::shared_ptr<Node>> &as_list();

        // const std::vector<std::shared_ptr<Node>> &as_list() const;

        std::map<std::string, std::shared_ptr<Node>> &as_object();

        // const std::map<std::string, std::shared_ptr<Node>> &as_object() const;
    };

} // namespace sp
