#include "Entry.h"
#include "Node.h"

//----------------------------------------------------------------------------------------------------------
// Entry
//----------------------------------------------------------------------------------------------------------
namespace sp
{

    Entry::Entry() : m_self_(nullptr) {}
    Entry::Entry(Entry const &other) : m_self_(other.m_self_) {}
    Entry::Entry(Entry &&other) : m_self_(other.m_self_) { other.m_self_ = nullptr; }
    Entry::~Entry() {}
    void Entry::swap(Entry &other) { std::swap(m_self_, other.m_self_); }

    class Attributes
    {
    public:
        Attributes() {}
        Attributes(Attributes const &other) : m_attributes_(other.m_attributes_) {}
        Attributes(Attributes &&other) : m_attributes_(std::move(other.m_attributes_)) {}
        ~Attributes() {}

        Attributes *copy() const { return new Attributes(*this); }

        std::ostream &repr(std::ostream &os) const
        {
            for (auto const &item : m_attributes_)
            {
                std::cout << "\t @" << item.first
                          << " : " << std::any_cast<std::string>(item.second) << ","
                          << std::endl;
            }
            return os;
        }

        bool has_a(std::string const &key) const
        {
            return m_attributes_.find(key) != m_attributes_.end();
        }
        bool check(std::string const &key, std::any const &v) const
        {
            NOT_IMPLEMENTED;
            return has_a(key);
        }
        void erase(std::string const &key) { m_attributes_.erase(m_attributes_.find(key)); }
        std::any get(std::string const &key) const { return m_attributes_.at(key); }
        void set(std::string const &key, std::any const &v) { m_attributes_[key] = v; }

        std::map<std::string, std::any> m_attributes_;
    };

    struct Content
    {
        virtual std::type_info const &type_info() const { return typeid(Content); };
        virtual Content *copy() const = 0;
        virtual Content *move() = 0;
    };
    struct ContentScalar : public Content
    {
        std::any content;
        ContentScalar(std::any const &v = nullptr) : content(v) {}
        ContentScalar(ContentScalar const &other) : content(other.content) {}
        ContentScalar(ContentScalar &&other) : content(std::move(other.content)) {}
        std::type_info const &type_info() const { return typeid(ContentScalar); }
        Content *copy() const { return new ContentScalar{*this}; };
        Content *move() { return new ContentScalar{std::move(*this)}; };
    };
    struct ContentBlock : public Content
    {
        std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>> content;
        ContentBlock(std::shared_ptr<char> const &p, std::type_info const &t, std::vector<size_t> const &d) : content({p, t, d}) {}
        ContentBlock(ContentBlock const &other) : content(other.content) {}
        ContentBlock(ContentBlock &&other) : content(std::move(other.content)) {}
        std::type_info const &type_info() const { return typeid(ContentBlock); }
        Content *copy() const { return new ContentBlock{*this}; };
        Content *move() { return new ContentBlock{std::move(*this)}; };
    };
    struct ContentList : public Content
    {
        std::vector<std::shared_ptr<Node>> content;
        ContentList() {}
        ContentList(ContentList const &other) : content(other.content) {}
        ContentList(ContentList &&other) : content(std::move(other.content)) {}
        std::type_info const &type_info() const { return typeid(ContentList); }
        Content *copy() const { return new ContentList{*this}; };
        Content *move() { return new ContentList{std::move(*this)}; };
    };
    struct ContentObject : public Content
    {
        std::map<std::string, std::shared_ptr<Node>> content;
        ContentObject() {}
        ContentObject(ContentObject const &other) : content(other.content) {}
        ContentObject(ContentObject &&other) : content(std::move(other.content)) {}
        std::type_info const &type_info() const { return typeid(ContentObject); }
        Content *copy() const { return new ContentObject{*this}; };
        Content *move() { return new ContentObject{std::move(*this)}; };
    };

    class EntryInMemory : public Entry
    {
    public:
        typedef EntryInMemory this_type;

        EntryInMemory();

        EntryInMemory(EntryInMemory const &other);

        EntryInMemory(EntryInMemory &&other);

        ~EntryInMemory();

        EntryInMemory &operator=(EntryInMemory const &other);

        void swap(EntryInMemory &other);

        Entry *copy() const;

        Entry *move();

        std::ostream &repr(std::ostream &os) const;

        int type() const;
        //----------------------------------------------------------------------------------------------------------
        // attribute
        //----------------------------------------------------------------------------------------------------------
        bool has_attribute(std::string const &k) const; // if key exists then return true else return false

        bool check_attribute(std::string const &k, std::any const &v) const; // if key exists and value ==v then return true else return false

        std::any get_attribute(std::string const &key) const; // get attribute at key, if key does not exist return nullptr

        void set_attribute(std::string const &key, std::any const &v); // set attribute at key as v

        void remove_attribute(std::string const &key = ""); // remove attribute at key, if key=="" then remove all

        // Range<Iterator<std::pair<std::string, std::any>>> attributes() const; // return reference of  all attributes

        //----------------------------------------------------------------------------------------------------------
        // as leaf node,  need node.type = Scalar || Block
        //----------------------------------------------------------------------------------------------------------
        typedef std::tuple<std::shared_ptr<char> /*data pointer*/, int /*element size*/, std::vector<size_t> /*dimensions*/> block_type;

        std::any get_scalar() const;

        void set_scalar(std::any const &);

        std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>> get_raw_block() const; // get block

        void set_raw_block(std::shared_ptr<char> const &, std::type_info const &, std::vector<size_t> const &); // set block

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        // function level 0
        void insert(std::string const &key, std::shared_ptr<Node> const &n = nullptr);

        std::shared_ptr<Node> find_child(std::string const &);

        std::shared_ptr<Node> child(std::string const &);

        std::shared_ptr<Node> child(int idx);

        std::shared_ptr<Node> append();

        void remove_child(int idx);

        void remove_child(std::string const &key);

        void remove_children();

        std::pair<Iterator<Node *>, Iterator<Node *>> children() const;

        // level 1
        std::pair<Iterator<Node *>, Iterator<Node *>> select(XPath const &path) const;

        std::shared_ptr<Node> select_one(XPath const &path) const;

    private:
        std::unique_ptr<Attributes> m_attributes_;
        std::unique_ptr<Content> m_content_;

        std::vector<std::shared_ptr<Node>> &as_list();
        const std::vector<std::shared_ptr<Node>> &as_list() const;
        std::map<std::string, std::shared_ptr<Node>> &as_object();
        const std::map<std::string, std::shared_ptr<Node>> &as_object() const;
    };
} // namespace sp
using namespace sp;

EntryInMemory::EntryInMemory()
    : Entry(),
      m_attributes_(new sp::Attributes{}),
      m_content_(new sp::ContentScalar{nullptr})
{
}

EntryInMemory::EntryInMemory(EntryInMemory const &other)
    : Entry(other),
      m_attributes_(other.m_attributes_->copy()),
      m_content_(other.m_content_->copy())
{
}

EntryInMemory::EntryInMemory(EntryInMemory &&other)
    : Entry(other),
      m_attributes_(std::move(other.m_attributes_)),
      m_content_(std::move(other.m_content_))
{
}

EntryInMemory::~EntryInMemory() {}

void EntryInMemory::swap(EntryInMemory &other)
{
    Entry::swap(other);
    std::swap(m_content_, other.m_content_);
    std::swap(m_attributes_, other.m_attributes_);
}

EntryInMemory &EntryInMemory::operator=(EntryInMemory const &other)
{
    EntryInMemory(other).swap(*this);
    return *this;
}

Entry *EntryInMemory::copy() const { return dynamic_cast<Entry *>(new EntryInMemory(*this)); };

Entry *EntryInMemory::move() { return dynamic_cast<Entry *>(new EntryInMemory(std::move(*this))); };

std::ostream &EntryInMemory::repr(std::ostream &os) const { return os; }

int EntryInMemory::type() const
{
    auto const &typeinfo = m_content_->type_info();

    NodeTag tag;

    if (typeinfo == typeid(ContentScalar))
    {
        tag = NodeTag::Scalar;
    }
    else if (typeinfo == typeid(ContentBlock))
    {
        tag = NodeTag::Block;
    }
    else if (typeinfo == typeid(ContentList))
    {
        tag = NodeTag::List;
    }
    else if (typeinfo == typeid(ContentObject))
    {
        tag = NodeTag::Object;
    }
    else
    {
        tag = NodeTag::Null;
    }
    return tag;
}

//----------------------------------------------------------------------------------------------------------
// attribute
//----------------------------------------------------------------------------------------------------------
bool EntryInMemory::has_attribute(std::string const &key) const
{
    return m_attributes_->has_a(key);
}

bool EntryInMemory::check_attribute(std::string const &key, std::any const &v) const
{
    return m_attributes_->check(key, v);
}

std::any EntryInMemory::get_attribute(std::string const &key) const
{
    return m_attributes_->get(key);
}

void EntryInMemory::set_attribute(std::string const &key, std::any const &v)
{
    m_attributes_->set(key, v);
}

void EntryInMemory::remove_attribute(std::string const &key)
{
    m_attributes_->erase(key);
}

// Range<Iterator<std::pair<std::string, std::any>>> EntryInMemory::attributes() const
// {
//     auto b = m_attributes_->m_attributes_.cbegin();
//     auto e = m_attributes_->m_attributes_.cend();
//     auto next = [=]() {
//         static auto it = b;
//         if (b == e)
//         {
//             return std::pair<std::string, std::any>{"", nullptr};
//         }
//         else
//         {
//             auto res = *it;
//             ++it;
//             return std::pair<std::string, std::any>{std::get<0>(res), std::get<1>(res)};
//         }
//     };
//     typedef Iterator<std::pair<std::string, std::any>> it;
//     return Range<it>{it(next), it(nullptr)};
// }

//----------------------------------------------------------------------------------------------------------
// as leaf node,  need node.type = Scalar || Block
//----------------------------------------------------------------------------------------------------------

std::any EntryInMemory::get_scalar() const
{
    if (m_content_->type_info() != typeid(ContentScalar))
    {
        throw std::runtime_error(std::string("Illegal type! [") + m_content_->type_info().name() + " != Scalar ]");
    }
    return dynamic_cast<ContentScalar const *>(m_content_.get())->content;
}

void EntryInMemory::set_scalar(std::any const &v)
{
    if (m_content_->type_info() == typeid(ContentList) || m_content_->type_info() == typeid(ContentObject))
    {
        throw std::runtime_error("Can not set value to tree node!");
    }
    m_content_.reset(new ContentScalar{v});
}

std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>> 
EntryInMemory::get_raw_block() const
{
    if (m_content_->type_info() != typeid(ContentBlock))
    {
        throw std::runtime_error(std::string("Illegal type! [") + m_content_->type_info().name() + " != Block ]");
    }
    return dynamic_cast<ContentBlock const *>(m_content_.get())->content;
}

void EntryInMemory::set_raw_block(std::shared_ptr<char> const &p, std::type_info const &t, std::vector<size_t> const &d)
{

    if (m_content_->type_info() == typeid(ContentList) || m_content_->type_info() == typeid(ContentObject))
    {
        throw std::runtime_error("Can not set value to tree node!");
    }
    m_content_.reset(new ContentBlock{p, t, d});
}

//----------------------------------------------------------------------------------------------------------
// convert
//----------------------------------------------------------------------------------------------------------

std::vector<std::shared_ptr<Node>> &
EntryInMemory::as_list()
{
    if (m_content_->type_info() != typeid(ContentList))
    {
        auto *p = new ContentList{};
        p->content.push_back(std::make_shared<Node>(m_self_, this->move()));
        m_content_.reset(dynamic_cast<Content *>(p));
    }
    return dynamic_cast<ContentList *>(m_content_.get())->content;
}

std::map<std::string, std::shared_ptr<Node>> &
EntryInMemory::as_object()
{
    if (m_content_->type_info() == typeid(Content))
    {
        m_content_.reset(dynamic_cast<Content *>(new ContentObject{}));
    }
    else if (m_content_->type_info() != typeid(ContentObject))
    {
        auto *p = new ContentObject{};
        p->content.emplace("_", std::make_shared<Node>(m_self_, this->move()));
        m_content_.reset(dynamic_cast<Content *>(p));
    }

    return dynamic_cast<ContentObject *>(m_content_.get())->content;
}

const std::vector<std::shared_ptr<Node>> &
EntryInMemory::as_list() const
{
    if (m_content_->type_info() != typeid(ContentList))
    {
        throw std::runtime_error("This is not a List Node");
    }
    return dynamic_cast<ContentList const *>(m_content_.get())->content;
}

const std::map<std::string, std::shared_ptr<Node>> &
EntryInMemory::as_object() const
{
    if (m_content_->type_info() != typeid(ContentObject))
    {
        throw std::runtime_error("This is not a List Node");
    }
    return dynamic_cast<ContentObject const *>(m_content_.get())->content;
}

//----------------------------------------------------------------------------------------------------------
// as Hierarchy tree node
// function level 0
void EntryInMemory::insert(std::string const &key, std::shared_ptr<Node> const &n)
{
    NOT_IMPLEMENTED;
}

std::shared_ptr<Node> EntryInMemory::find_child(std::string const &)
{
    NOT_IMPLEMENTED;
    return nullptr;
}

std::shared_ptr<Node> EntryInMemory::child(std::string const &key)
{
    auto &m = as_object();
    auto p = m.find(key);
    if (p == m.end())
    {
        auto n = std::make_shared<Node>(m_self_, new EntryInMemory);
        n->set_attribute<std::string>("@name", key);
        return m.emplace(key, n).first->second;
    }
    else
    {
        return p->second;
    }
}

std::shared_ptr<Node> EntryInMemory::child(int idx) { return as_list()[idx]; }

std::shared_ptr<Node> EntryInMemory::append()
{
    auto &l = as_list();
    auto n = std::make_shared<Node>(m_self_, new EntryInMemory);
    n->set_attribute<int>("@id", size(l));
    l.push_back(n);
    return n;
}

void EntryInMemory::remove_child(int idx)
{
    NOT_IMPLEMENTED;
}

void EntryInMemory::remove_child(std::string const &key)
{
    if (m_content_->type_info() == typeid(ContentObject))
    {
        auto &m = as_object();
        m.erase(m.find(key));
    }
}

void EntryInMemory::remove_children() { m_content_.reset(new ContentScalar{}); }

std::pair<Iterator<Node *>, Iterator<Node *>>
EntryInMemory::children() const
{
    if (m_content_->type_info() == typeid(ContentList))
    {
        auto const &m = as_list();
        auto b = m.begin();
        auto e = m.end();
        return std::make_pair(Iterator<Node *>(b, [](std::remove_reference_t<decltype(b)> const &p) { return p->get(); }), Iterator<Node *>(e));
    }
    else if (m_content_->type_info() == typeid(ContentObject))
    {
        auto const &m = as_object();
        auto b = m.begin();
        auto e = m.end();

        return std::make_pair(Iterator<Node *>(b, [](std::remove_reference_t<decltype(b)> const &p) { return p->second.get(); }), Iterator<Node *>(e));
    }
    else
    {
        return std::make_pair(Iterator<Node *>(),Iterator<Node *>());
    }
}

std::pair<Iterator<Node *>, Iterator<Node *>>
EntryInMemory::select(XPath const &path) const
{
    NOT_IMPLEMENTED;
    return std::pair<Iterator<Node *>, Iterator<Node *>>{};
}

std::shared_ptr<Node> EntryInMemory::select_one(XPath const &path) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

Entry *Entry::create(std::string const &backend)
{
    return new EntryInMemory();
}
