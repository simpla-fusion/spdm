#include "Entry.h"
#include "Node.h"
#include "utility/Logger.h"

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

        void swap(Attributes &other)
        {
            std::swap(m_attributes_, other.m_attributes_);
        }

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

        bool has_a(std::string const &key) const { return m_attributes_.find(key) != m_attributes_.end(); }

        bool check(std::string const &key, std::any const &v) const
        {
            NOT_IMPLEMENTED;
            return has_a(key);
        }

        void erase(std::string const &key) { m_attributes_.erase(m_attributes_.find(key)); }

        std::any get(std::string const &key) const { return m_attributes_.at(key); }

        std::any get(std::string const &key, std::any const &default_value)
        {
            return m_attributes_.emplace(key, default_value).first->second;
        }

        void set(std::string const &key, std::any const &v) { m_attributes_[key] = v; }

        Range<Iterator<const std::pair<const std::string, std::any>>>
        items() const
        {
            return std::move(Range<Iterator<const std::pair<const std::string, std::any>>>{
                Iterator<const std::pair<const std::string, std::any>>{m_attributes_.begin(), [](const auto &it) { return it.operator->(); }},
                Iterator<const std::pair<const std::string, std::any>>{m_attributes_.end()}});
        }

        std::map<std::string, std::any> m_attributes_;
    };

    struct Content
    {
        virtual std::type_info const &type_info() const { return typeid(Content); };
        virtual Content *copy() const { return nullptr; }
        virtual Content *move() { return nullptr; }
    };

    struct ContentScalar : public Content
    {
        std::any content;
        ContentScalar() {}
        ContentScalar(const ContentScalar &other) : content(other.content) {}
        ContentScalar(ContentScalar &&other) : content(std::move(other.content)) {}
        std::type_info const &type_info() const { return typeid(ContentScalar); }
        Content *copy() const override { return new ContentScalar(*this); };
        Content *move() override { return new ContentScalar(std::move(*this)); };
    };

    struct ContentBlock : public Content
    {
        std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>> content;
        ContentBlock(std::shared_ptr<char> const &p, std::type_info const &t, std::vector<size_t> const &d) : content({p, t, d}) {}
        ContentBlock(const ContentBlock &other) : content(other.content) {}
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

    struct entry_tag_memory
    {
        std::unique_ptr<Attributes> m_attributes_;
        std::unique_ptr<Content> m_content_;

        entry_tag_memory *copy() const
        {
            auto *p = new entry_tag_memory;
            p->m_attributes_.reset(new Attributes(*m_attributes_));
            p->m_content_.reset(new Content);
            return p;
        }

        std::vector<std::shared_ptr<Node>> &as_list();
        // const std::vector<std::shared_ptr<Node>> &as_list() const;
        std::map<std::string, std::shared_ptr<Node>> &as_object();
        // const std::map<std::string, std::shared_ptr<Node>> &as_object() const;
    };
    template <>
    EntryTmpl<entry_tag_memory>::EntryTmpl() : Entry(), m_pimpl_(new entry_tag_memory) {}
    template <>
    EntryTmpl<entry_tag_memory>::EntryTmpl(EntryTmpl<entry_tag_memory> const &other)
        : Entry(other), m_pimpl_(other.m_pimpl_->copy()) {}
    template <>
    EntryTmpl<entry_tag_memory>::EntryTmpl(EntryTmpl<entry_tag_memory> &&other)
        : Entry(other), m_pimpl_(other.m_pimpl_.release()) {}

    template <>
    EntryTmpl<entry_tag_memory>::~EntryTmpl() {}
    template <>
    void EntryTmpl<entry_tag_memory>::swap(EntryTmpl<entry_tag_memory> &other)
    {
        Entry::swap(other);
        std::swap(m_pimpl_, other.m_pimpl_);
    }
    template <>
    EntryTmpl<entry_tag_memory> &EntryTmpl<entry_tag_memory>::operator=(EntryTmpl<entry_tag_memory> const &other)
    {
        EntryTmpl<entry_tag_memory>(other).swap(*this);
        return *this;
    }
    template <>
    Entry *EntryTmpl<entry_tag_memory>::copy() const { return dynamic_cast<Entry *>(new EntryTmpl<entry_tag_memory>(*this)); };
    template <>
    Entry *EntryTmpl<entry_tag_memory>::move()
    {
        auto res = new EntryTmpl<entry_tag_memory>();
        res->swap(*this);
        return res;
    };
    template <>
    std::ostream &EntryTmpl<entry_tag_memory>::repr(std::ostream &os) const { return os; }
    template <>
    int EntryTmpl<entry_tag_memory>::type() const
    {
        auto const &typeinfo = m_pimpl_->m_content_->type_info();

        NodeTag tag;
        if (typeinfo == typeid(Content))
        {
            tag = NodeTag::Null;
        }

        else if (typeinfo == typeid(ContentScalar))
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
    template <>
    bool EntryTmpl<entry_tag_memory>::has_attribute(std::string const &key) const { return m_pimpl_->m_attributes_->has_a(key); }
    template <>
    bool EntryTmpl<entry_tag_memory>::check_attribute(std::string const &key, std::any const &v) const { return m_pimpl_->m_attributes_->check(key, v); }
    template <>
    std::any EntryTmpl<entry_tag_memory>::get_attribute(std::string const &key) const
    {
        return m_pimpl_->m_attributes_->get(key);
    }
    template <>
    std::any EntryTmpl<entry_tag_memory>::get_attribute(std::string const &key, std::any const &default_value)
    {
        return m_pimpl_->m_attributes_->get(key, default_value);
    }
    template <>
    void EntryTmpl<entry_tag_memory>::set_attribute(std::string const &key, std::any const &v)
    {
        m_pimpl_->m_attributes_->set(key, v);
    }
    template <>
    void EntryTmpl<entry_tag_memory>::remove_attribute(std::string const &key) { m_pimpl_->m_attributes_->erase(key); }
    template <>
    Range<Iterator<const std::pair<const std::string, std::any>>>
    EntryTmpl<entry_tag_memory>::attributes() const { return std::move(m_pimpl_->m_attributes_->items()); }

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    template <>
    std::any EntryTmpl<entry_tag_memory>::get_scalar() const
    {
        if (m_pimpl_->m_content_->type_info() != typeid(ContentScalar))
        {
            throw std::runtime_error(std::string("Illegal type! [") + m_pimpl_->m_content_->type_info().name() + " != Scalar ]");
        }
        return dynamic_cast<ContentScalar const *>(m_pimpl_->m_content_.get())->content;
    }
    template <>
    void EntryTmpl<entry_tag_memory>::set_scalar(std::any const &v)
    {
        if (m_pimpl_->m_content_->type_info() == typeid(ContentList) || m_pimpl_->m_content_->type_info() == typeid(ContentObject))
        {
            throw std::runtime_error("Can not set value to tree node!");
        }
        auto *p = new ContentScalar{};
        std::any(v).swap(p->content);
        m_pimpl_->m_content_.reset(p);
    }
    template <>
    std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>>
    EntryTmpl<entry_tag_memory>::get_raw_block() const
    {
        if (m_pimpl_->m_content_->type_info() != typeid(ContentBlock))
        {
            throw std::runtime_error(std::string("Illegal type! [") + m_pimpl_->m_content_->type_info().name() + " != Block ]");
        }
        return dynamic_cast<ContentBlock const *>(m_pimpl_->m_content_.get())->content;
    }
    template <>
    void EntryTmpl<entry_tag_memory>::set_raw_block(std::shared_ptr<char> const &p, std::type_info const &t, std::vector<size_t> const &d)
    {

        if (m_pimpl_->m_content_->type_info() == typeid(ContentList) || m_pimpl_->m_content_->type_info() == typeid(ContentObject))
        {
            throw std::runtime_error("Can not set value to tree node!");
        }
        m_pimpl_->m_content_.reset(new ContentBlock{p, t, d});
    }

    //----------------------------------------------------------------------------------------------------------
    // convert
    //----------------------------------------------------------------------------------------------------------

    std::vector<std::shared_ptr<Node>> &
    entry_tag_memory::as_list()
    {
        if (m_content_->type_info() == typeid(Content))
        {
            m_content_.reset(dynamic_cast<Content *>(new ContentList{}));
        }
        else if (m_content_->type_info() != typeid(ContentList))
        {
            auto *p = new ContentList{};
            auto n = std::make_shared<Node>(); //(m_self_, this->move());
            p->content.push_back(n);
            if (n->has_attribute("@name"))
            {
                m_attributes_->set("@name", n->get_attribute<std::string>("@name"));
            }
            else
            {
                m_attributes_->set("@name", "_");
            }

            m_content_.reset(dynamic_cast<Content *>(p));
        }
        return dynamic_cast<ContentList *>(m_content_.get())->content;
    }

    std::map<std::string, std::shared_ptr<Node>> &
    entry_tag_memory::as_object()
    {
        if (m_content_->type_info() == typeid(Content))
        {
            m_content_.reset(dynamic_cast<Content *>(new ContentObject{}));
        }
        else if (m_content_->type_info() != typeid(ContentObject))
        {
            auto *p = new ContentObject{};
            // p->content.emplace("_", std::make_shared<Node>(m_self_, this->move()));
            m_content_.reset(dynamic_cast<Content *>(p));
        }

        return dynamic_cast<ContentObject *>(m_content_.get())->content;
    }

    // const std::vector<std::shared_ptr<Node>> &
    // EntryTmpl<entry_tag_memory>::as_list() const
    // {
    //     if (m_pimpl_->m_content_->type_info() != typeid(ContentList))
    //     {
    //         throw std::runtime_error("This is not a List Node");
    //     }
    //     return dynamic_cast<ContentList const *>(m_pimpl_->m_content_.get())->content;
    // }

    // const std::map<std::string, std::shared_ptr<Node>> &
    // EntryTmpl<entry_tag_memory>::as_object() const
    // {
    //     if (m_pimpl_->m_content_->type_info() != typeid(ContentObject))
    //     {
    //         throw std::runtime_error("This is not a List Node");
    //     }
    //     return dynamic_cast<ContentObject const *>(m_pimpl_->m_content_.get())->content;
    // }

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0
    template <>
    std::shared_ptr<Node>
    EntryTmpl<entry_tag_memory>::insert(std::string const &key, std::shared_ptr<Node> const &n)
    {
        NOT_IMPLEMENTED;
        return nullptr;
    }
    template <>
    std::shared_ptr<Node>
    EntryTmpl<entry_tag_memory>::find_child(std::string const &)
    {
        NOT_IMPLEMENTED;
        return nullptr;
    }
    template <>
    std::shared_ptr<const Node>
    EntryTmpl<entry_tag_memory>::find_child(std::string const &) const
    {
        NOT_IMPLEMENTED;
        return nullptr;
    }
    template <>
    std::shared_ptr<Node>
    EntryTmpl<entry_tag_memory>::child(std::string const &key)
    {
        auto &m = m_pimpl_->as_object();
        auto p = m.find(key);
        if (p == m.end())
        {
            auto n = std::make_shared<Node>(m_self_, new EntryTmpl<entry_tag_memory>);
            n->set_attribute<std::string>("@name", key);
            return m.emplace(key, n).first->second;
        }
        else
        {
            return p->second;
        }
    }

    // std::shared_ptr<const Node> EntryTmpl<entry_tag_memory>::child(std::string const &key) const
    // {
    //     auto p = find_child(key);
    //     if (p == nullptr)
    //     {
    //         throw std::runtime_error(std::string("Can not find " + key));
    //     }
    //     return p;
    // }
    template <>
    std::shared_ptr<Node>
    EntryTmpl<entry_tag_memory>::child(int idx) { return m_pimpl_->as_list()[idx]; }

    // std::shared_ptr<const Node> EntryTmpl<entry_tag_memory>::child(int idx) const { return m_pimpl_->as_list()[idx]; }
    template <>
    std::shared_ptr<Node>
    EntryTmpl<entry_tag_memory>::append()
    {
        auto &l = m_pimpl_->as_list();
        auto n = std::make_shared<Node>(m_self_, new EntryTmpl<entry_tag_memory>);
        n->set_attribute<int>("@id", size(l));
        l.push_back(n);
        return n;
    }
    template <>
    std::shared_ptr<Node>
    EntryTmpl<entry_tag_memory>::append(std::shared_ptr<Node> const &n)
    {
        auto &l = m_pimpl_->as_list();
        n->set_attribute<int>("@id", l.size());
        l.push_back(n);
        return n;
    }
    template <>
    void EntryTmpl<entry_tag_memory>::append(const Iterator<std::shared_ptr<Node>> &b, const Iterator<std::shared_ptr<Node>> &e)
    {
        for (auto it = b; b != e; ++it)
        {
            append(*it);
        }
    }
    template <>
    void EntryTmpl<entry_tag_memory>::insert(Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &b,
                                             Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &e)
    {
        for (auto it = b; b != e; ++it)
        {
            insert(it->first, it->second);
        }
    }
    template <>
    void EntryTmpl<entry_tag_memory>::remove_child(int idx)
    {
        NOT_IMPLEMENTED;
    }
    template <>
    void EntryTmpl<entry_tag_memory>::remove_child(std::string const &key)
    {
        if (m_pimpl_->m_content_->type_info() == typeid(ContentObject))
        {
            auto &m = m_pimpl_->as_object();
            m.erase(m.find(key));
        }
    }
    template <>
    void EntryTmpl<entry_tag_memory>::remove_children() { m_pimpl_->m_content_.reset(new Content{}); }

    // std::pair<Iterator<const Node>, Iterator<const Node>>
    // EntryTmpl<entry_tag_memory>::children() const
    // {
    //     if (m_pimpl_->m_content_->type_info() == typeid(ContentList))
    //     {
    //         auto const &m = m_pimpl_->as_list();
    //         auto b = m.begin();
    //         auto e = m.end();
    //         return std::move(std::make_pair(
    //             Iterator<const Node>(b, [](auto const &p) { return p->get(); }),
    //             Iterator<const Node>(e)));
    //     }
    //     else if (m_pimpl_->m_content_->type_info() == typeid(ContentObject))
    //     {
    //         auto const &m = m_pimpl_->as_object();
    //         auto b = m.begin();
    //         auto e = m.end();

    //         return std::move(std::make_pair(
    //             Iterator<const Node>(b, [](auto const &p) { return p->second.get(); }),
    //             Iterator<const Node>(e)));
    //     }
    //     else
    //     {
    //         return std::move(std::make_pair(Iterator<const Node>(), Iterator<const Node>()));
    //     }
    // }
    template <>
    std::pair<Iterator<Node>, Iterator<Node>>
    EntryTmpl<entry_tag_memory>::children()
    {
        if (m_pimpl_->m_content_->type_info() == typeid(ContentList))
        {
            auto const &m = m_pimpl_->as_list();
            auto b = m.begin();
            auto e = m.end();
            return std::move(std::make_pair(
                Iterator<Node>(b, [](auto const &p) { return p->get(); }),
                Iterator<Node>(e)));
        }
        else if (m_pimpl_->m_content_->type_info() == typeid(ContentObject))
        {
            auto const &m = m_pimpl_->as_object();
            auto b = m.begin();
            auto e = m.end();

            return std::move(std::make_pair(
                Iterator<Node>(b, [](auto const &p) { return p->second.get(); }),
                Iterator<Node>(e)));
        }
        else
        {
            return std::move(std::make_pair(Iterator<Node>(), Iterator<Node>()));
        }
    }
    template <>
    std::pair<Iterator<Node>, Iterator<Node>>
    EntryTmpl<entry_tag_memory>::select(XPath const &path)
    {
        NOT_IMPLEMENTED;
        return std::pair<Iterator<const Node>, Iterator<const Node>>{};
    }

    // std::pair<Iterator<const Node>, Iterator<const Node>>
    // EntryTmpl<entry_tag_memory>::select(XPath const &path) const
    // {
    //     NOT_IMPLEMENTED;
    //     return std::pair<Iterator<const Node>, Iterator<const Node>>{};
    // }

    // std::shared_ptr<const Node> EntryTmpl<entry_tag_memory>::select_one(XPath const &path) const { return child(path.str()); }
    template <>
    std::shared_ptr<Node> EntryTmpl<entry_tag_memory>::select_one(XPath const &path) { return child(path.str()); }

    //#######################################################################################################################################
    Entry *Entry::create(std::string const &backend)
    {
        if (backend == "XML")
        {
            return new EntryTmpl<entry_tag_memory>();
        }
        else
        {
            return new EntryTmpl<entry_tag_memory>();
        }
    }

} // namespace sp
