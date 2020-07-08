#include "EntryImpl.h"
#include "Node.h"
#include <any>
#include <map>
#include <vector>

//----------------------------------------------------------------------------------------------------------
// Attributes
//----------------------------------------------------------------------------------------------------------
namespace sp
{

    template <>
    class EntryImpl<nullptr_t> : public Entry
    {
    public:
        typedef EntryImpl<nullptr_t> this_type;
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

        Entry *copy() const;

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

        void as_scalar(std::any const &);

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

    private:
        std::map<std::string, std::any> m_attributes_;
        std::any m_content_;
        NodeTag m_node_type_;
    };
} // namespace sp
using namespace sp;

EntryImpl<nullptr_t>::EntryImpl() {}

EntryImpl<nullptr_t>::EntryImpl(this_type const &other)
    : m_attributes_(other.m_attributes_),
      m_content_(other.m_content_),
      m_node_type_(other.m_node_type_)
{
}

EntryImpl<nullptr_t>::EntryImpl(this_type &&other)
    : m_attributes_(std::move(other.m_attributes_)),
      m_content_(std::move(other.m_content_)),
      m_node_type_(other.m_node_type_)
{
}

EntryImpl<nullptr_t>::~EntryImpl() {}

void EntryImpl<nullptr_t>::swap(this_type &other)
{
    std::swap(m_content_, other.m_content_);
    std::swap(m_attributes_, other.m_attributes_);
    std::swap(m_node_type_, other.m_node_type_);
}

Entry *EntryImpl<nullptr_t>::copy() const { return new this_type(*this); };

std::ostream &EntryImpl<nullptr_t>::repr(std::ostream &os) const
{
    NOT_IMPLEMENTED;
    return os;
}

int EntryImpl<nullptr_t>::type() const { return m_node_type_; }

//----------------------------------------------------------------------------------------------------------
// attribute
//----------------------------------------------------------------------------------------------------------
bool EntryImpl<nullptr_t>::has_attribute(std::string const &key) const { return m_attributes_.find(key) != m_attributes_.end(); }
bool EntryImpl<nullptr_t>::check_attribute(std::string const &key, std::any const &v) const
{
    return has_attribute(key) &&
           std::any_cast<std::string>(m_attributes_.at(key)) == std::any_cast<std::string>(v);
}
std::any EntryImpl<nullptr_t>::attribute(std::string const &key) const { return m_attributes_.at(key); }
void EntryImpl<nullptr_t>::attribute(std::string const &key, std::any const &v) { m_attributes_[key] = v; }
void EntryImpl<nullptr_t>::remove_attribute(std::string const &key) { m_attributes_.erase(m_attributes_.find(key)); }
Range<Iterator<std::pair<std::string, std::any>>> EntryImpl<nullptr_t>::attributes() const { return Range<Iterator<std::pair<std::string, std::any>>>{}; }
//----------------------------------------------------------------------------------------------------------
// as leaf node,  need node.type = Scalar || Block
//----------------------------------------------------------------------------------------------------------

std::any EntryImpl<nullptr_t>::as_scalar() const { return m_content_; }

void EntryImpl<nullptr_t>::as_scalar(std::any const &v) { m_content_ = v; }

EntryImpl<nullptr_t>::block_type EntryImpl<nullptr_t>::as_block() const { return std::any_cast<block_type>(m_content_); }

void EntryImpl<nullptr_t>::as_block(block_type const &blk) { m_content_ = std::any(blk); }

Entry &EntryImpl<nullptr_t>::as_list() { return *this; }

Entry &EntryImpl<nullptr_t>::as_object() { return *this; }
//----------------------------------------------------------------------------------------------------------
// as Hierarchy tree node
// function level 0

std::shared_ptr<Node> EntryImpl<nullptr_t>::child(std::string const &)
{
    NOT_IMPLEMENTED;
    return nullptr;
}

std::shared_ptr<const Node> EntryImpl<nullptr_t>::child(std::string const &) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

std::shared_ptr<Node> EntryImpl<nullptr_t>::child(int idx)
{
    NOT_IMPLEMENTED;
    return nullptr;
}

std::shared_ptr<const Node> EntryImpl<nullptr_t>::child(int idx) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

std::shared_ptr<Node> EntryImpl<nullptr_t>::append()
{
    NOT_IMPLEMENTED;
    return nullptr;
}

void EntryImpl<nullptr_t>::remove_child(int idx)
{
    NOT_IMPLEMENTED;
}

void EntryImpl<nullptr_t>::remove_child(std::string const &key)
{
    NOT_IMPLEMENTED;
}

void EntryImpl<nullptr_t>::remove_children()
{
    NOT_IMPLEMENTED;
}

Range<Iterator<std::shared_ptr<Node>>> EntryImpl<nullptr_t>::children() const
{
    NOT_IMPLEMENTED;
    return Range<Iterator<std::shared_ptr<Node>>>{};
}

Range<Iterator<std::shared_ptr<Node>>> EntryImpl<nullptr_t>::select(XPath const &path) const
{
    NOT_IMPLEMENTED;
    return Range<Iterator<std::shared_ptr<Node>>>{};
}

std::shared_ptr<Node> EntryImpl<nullptr_t>::select_one(XPath const &path) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}
