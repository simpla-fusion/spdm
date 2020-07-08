#include "Entry.h"
#include <string>
namespace sp
{
    template <typename BACKEND_TAG>
    class EntryImpl : public Entry
    {
    public:
        typedef EntryImpl<BACKEND_TAG> this_type;

        EntryImpl();

        EntryImpl(this_type const &other);

        EntryImpl(this_type &&other);

        ~EntryImpl();

        this_type &operator=(this_type const &other)
        {
            this_type(other).swap(*this);
            return *this;
        };

        void swap(this_type &other);

        Entry *copy() const { return new this_type(*this); };

        std::ostream &repr(std::ostream &os) const;

        int type() const;
        //----------------------------------------------------------------------------------------------------------
        // attribute
        //----------------------------------------------------------------------------------------------------------
        bool has_attribute(std::string const &k) const;                       // if key exists then return true else return false
        bool check_attribute(std::string const &k, std::any const &v) const;  // if key exists and value ==v then return true else return false
        std::any attribute(std::string const &key) const;                     // get attribute at key, if key does not exist return nullptr
        void attribute(std::string const &key, std::any const &v);            // set attribute at key as v
        void remove_attribute(std::string const &key = "");                   // remove attribute at key, if key=="" then remove all
        Range<Iterator<std::pair<std::string, std::any>>> attributes() const; // return reference of  all attributes

        //----------------------------------------------------------------------------------------------------------
        // as leaf node,  need node.type = Scalar || Block
        //----------------------------------------------------------------------------------------------------------
        typedef std::tuple<std::shared_ptr<char> /*data pointer*/, int /*element size*/, std::vector<size_t> /*dimensions*/> block_type;

        std::any as_scalar() const;

        void as_scalar(std::any);

        block_type as_block() const;

        void as_block(block_type const &);

        Entry &as_list();

        Entry &as_object();
        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        // function level 0
        std::shared_ptr<Node> child(std::string const &);

        std::shared_ptr<const Node> child(std::string const &) const;

        std::shared_ptr<Node> child(int idx);

        std::shared_ptr<const Node> child(int idx) const;

        std::shared_ptr<Node> append();

        void remove_child(int idx);

        void remove_child(std::string const &key);

        void remove_children();

        Range<Iterator<std::shared_ptr<Node>>> children() const;

        // level 1
        Range<Iterator<std::shared_ptr<Node>>> select(XPath const &path) const;

        std::shared_ptr<Node> select_one(XPath const &path) const;
    };
  
} // namespace sp