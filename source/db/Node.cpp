#include "Node.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
#include "Entry.h"

namespace sp::utility
{

std::ostream& fancy_print(std::ostream& os, const sp::db::tree_node_type& v, int indent = 0, int tab = 4);

} // namespace sp::utility

namespace sp::db
{

std::ostream& operator<<(std::ostream& os, tree_node_type const& entry) { return sp::utility::fancy_print(os, entry, 0); }

//----------------------------------------------------------------------------------------------------
std::pair<std::shared_ptr<const NodeObject>, Path> NodeObject::full_path() const
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<const NodeObject>, Path>{shared_from_this(), {}};
}

std::pair<std::shared_ptr<NodeObject>, Path> NodeObject::full_path()
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<NodeObject>, Path>{shared_from_this(), {}};
}

void NodeObject::merge(const NodeObject&) { NOT_IMPLEMENTED; }

void NodeObject::patch(const NodeObject&) { NOT_IMPLEMENTED; }

void NodeObject::update(const NodeObject&) { NOT_IMPLEMENTED; }

bool NodeObject::compare(const tree_node_type& other) const
{
    NOT_IMPLEMENTED;
    return false;
}

tree_node_type NodeObject::diff(const tree_node_type& other) const
{
    NOT_IMPLEMENTED;
    return tree_node_type{};
}
//==========================================================================================
// NodeArray

std::shared_ptr<NodeArray> NodeArray::create(const std::string& backend) { return std::make_shared<NodeArray>(); }

size_t NodeArray::size() const { return m_container_.size(); }

void NodeArray::clear() { m_container_.clear(); }

Cursor<tree_node_type> NodeArray::children()
{
    NOT_IMPLEMENTED;
    return Cursor<tree_node_type>(); /*(m_container_.begin(), m_container_.end());*/
}

Cursor<const tree_node_type> NodeArray::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const tree_node_type>(); /*(m_container_.cbegin(), m_container_.cend());*/
}

void NodeArray::for_each(std::function<void(int, tree_node_type&)> const& visitor)
{
    for (int i = 0, s = m_container_.size(); i < s; ++i)
    {
        visitor(i, m_container_[i]);
    }
}

void NodeArray::for_each(std::function<void(int, const tree_node_type&)> const& visitor) const
{
    for (int i = 0, s = m_container_.size(); i < s; ++i)
    {
        visitor(i, m_container_[i]);
    }
}

tree_node_type NodeArray::slice(int start, int stop, int step)
{
    NOT_IMPLEMENTED;
    return tree_node_type{};
}

tree_node_type NodeArray::slice(int start, int stop, int step) const
{
    NOT_IMPLEMENTED;
    return tree_node_type{};
}

void NodeArray::resize(std::size_t num) { m_container_.resize(num); }

tree_node_type& NodeArray::insert(int idx, tree_node_type v)
{
    if (m_container_[idx].index() == tree_node_tags::Null)
    {
        m_container_[idx].swap(v);
    }
    return m_container_[idx];
}

const tree_node_type& NodeArray::at(int idx) const { return m_container_.at(idx); }

tree_node_type& NodeArray::at(int idx) { return m_container_.at(idx); }

tree_node_type& NodeArray::push_back(tree_node_type v)
{
    m_container_.emplace_back(std::move(v));
    return m_container_.back();
}

tree_node_type NodeArray::pop_back()
{
    tree_node_type res(m_container_.back());
    m_container_.pop_back();
    return std::move(res);
}

//----------------------------------------------------------------------------------------------------

class NodeObjectDefault : public NodeObject
{
private:
    std::map<std::string, tree_node_type> m_container_;

public:
    typedef NodeObjectDefault this_type;
    typedef tree_node_tags value_type_tags;

    NodeObjectDefault() = default;

    NodeObjectDefault(const this_type& other) : m_container_(other.m_container_) {}

    NodeObjectDefault(this_type&& other) : m_container_(std::move(other.m_container_)) {}

    virtual ~NodeObjectDefault() = default;

    std::unique_ptr<NodeObject> copy() const override { return std::unique_ptr<NodeObject>(new NodeObjectDefault(*this)); }

    void load(const std::string&) override {}

    void save(const std::string&) const override { NOT_IMPLEMENTED; }

    size_t size() const override { return m_container_.size(); }

    void clear() override { return m_container_.clear(); }

    // Entry at(Path path) override;

    // Entry at(Path path) const override;

    Cursor<tree_node_type> children() override;

    Cursor<const tree_node_type> children() const override;

    // void for_each(std::function<void(const std::string&, tree_node_type&)> const&) override;

    void for_each(std::function<void(const std::string&, const tree_node_type&)> const&) const override;
    //------------------------------------------------------------------

    tree_node_type insert(Path, tree_node_type) override;

    tree_node_type find(Path) const override;

    void update(Path, tree_node_type) override;

    void remove(Path) override;
};

Cursor<tree_node_type> NodeObjectDefault::children()
{
    return Cursor<tree_node_type>{};
}

Cursor<const tree_node_type> NodeObjectDefault::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const tree_node_type>{};
}

// void NodeObjectDefault::for_each(std::function<void(const std::string&, tree_node_type&)> const& visitor)
// {
//     for (auto&& item : m_container_)
//     {
//         visitor(item.first, item.second);
//     }
// }

void NodeObjectDefault::for_each(std::function<void(const std::string&, const tree_node_type&)> const& visitor) const
{
    for (auto&& item : m_container_)
    {
        visitor(item.first, item.second);
    }
}

namespace _detail
{

tree_node_type insert(tree_node_type self, Path::Segment path_seg, tree_node_type v)
{

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<tree_node_tags::Object, tree_node_type>& object_p) {
                object_p->insert(path_seg, tree_node_type{NodeObject::create()}).swap(self);
            },
            [&](std::variant_alternative_t<tree_node_tags::Array, tree_node_type>& array_p) {
                array_p->insert(std::get<Path::segment_tags::Index>(path_seg), tree_node_type{NodeObject::create()}).swap(self);
            },
            [&](std::variant_alternative_t<tree_node_tags::Block, tree_node_type>&) {
                NOT_IMPLEMENTED;
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
        self);

    return std::move(self);
}

tree_node_type insert(tree_node_type self, Path path, tree_node_type v)
{
    for (auto it = path.begin(), ie = --path.end(); it != ie; ++it)
    {
        insert(self, *it, tree_node_type{NodeObject::create()});
    }
    insert(self, path.last(), std::move(v)).swap(self);
    return self;
}
void update(tree_node_type self, Path path, tree_node_type v)
{
    for (auto it = path.begin(), ie = --path.end(); it != ie; ++it)
    {
        insert(self, *it, tree_node_type{NodeObject::create()}).swap(self);
    }

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<tree_node_tags::Object, tree_node_type>& object_p) {
                object_p->update(path.last(), std::move(v));
            },
            [&](std::variant_alternative_t<tree_node_tags::Array, tree_node_type>& array_p) {
                array_p->insert(std::get<Path::segment_tags::Index>(path.last()), std::move(v));
            },
            [&](std::variant_alternative_t<tree_node_tags::Block, tree_node_type>&) {
                NOT_IMPLEMENTED;
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
        self);
}

tree_node_type find(tree_node_type self, Path path)
{

    bool found = true;
    Path prefix;

    for (auto it = path.begin(), ie = path.end(); it != ie; ++it)
    {
        prefix.append(*it);

        switch (self.index())
        {
        case tree_node_tags::Object:
            tree_node_type(std::get<tree_node_tags::Object>(self)->find(*it)).swap(self);
            break;
        case tree_node_tags::Array:
            tree_node_type(std::get<tree_node_tags::Array>(self)->at(std::get<Path::segment_tags::Index>(*it))).swap(self);
            break;
        default:
            found = false;
            break;
        }
    }
    if (!found)
    {
        throw std::out_of_range("Can not find url:" + prefix.str());
    }

    return std::move(self);
}

void remove(tree_node_type self, Path path)
{
    find(self, path.prefix()).swap(self);

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<tree_node_tags::Object, tree_node_type>& object_p) {
                object_p->remove(path.last());
            },
            [&](std::variant_alternative_t<tree_node_tags::Array, tree_node_type>& array_p) {
                array_p->at(std::get<Path::segment_tags::Index>(path.last())).emplace<tree_node_tags::Null>(nullptr);
            },
            [&](std::variant_alternative_t<tree_node_tags::Block, tree_node_type>&) {
                NOT_IMPLEMENTED;
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
        self);
}

} // namespace _detail

tree_node_type NodeObjectDefault::insert(Path path, tree_node_type v)
{
    tree_node_type res;

    switch (path.length())
    {
    case 0:
        NOT_IMPLEMENTED;
        break;
    case 1:
        tree_node_type(m_container_.emplace(std::get<Path::segment_tags::Key>(*path.begin()), std::move(v)).first->second).swap(res);
        break;
    default:
        _detail::insert(tree_node_type{std::dynamic_pointer_cast<NodeObject>(shared_from_this())}, path, std::move(v)).swap(res);
        break;
    }
    return std::move(res);
}

tree_node_type NodeObjectDefault::find(Path path) const
{
    tree_node_type res;

    switch (path.length())
    {
    case 0:
        tree_node_type{std::dynamic_pointer_cast<NodeObject>(const_cast<NodeObjectDefault*>(this)->shared_from_this())}.swap(res);
        break;
    case 1:
        tree_node_type(m_container_.at(std::get<Path::segment_tags::Key>(*path.begin()))).swap(res);
        break;
    default:
        _detail::find(tree_node_type{}, path).swap(res);
    }

    return std::move(res);
}

void NodeObjectDefault::update(Path path, tree_node_type v)
{
    switch (path.length())
    {
    case 0:
        NOT_IMPLEMENTED;
        break;
    case 1:
        m_container_[std::get<Path::segment_tags::Key>(*path.begin())].swap(v);
        break;
    default:
        _detail::update(tree_node_type{std::dynamic_pointer_cast<NodeObject>(shared_from_this())}, path, std::move(v));
    }
}

void NodeObjectDefault::remove(Path path)
{
    tree_node_type self;
    switch (path.length())
    {
    case 0:
        m_container_.clear();
        break;
    case 1:
        m_container_.erase(std::get<Path::segment_tags::Key>(*path.begin()));
        break;
    default:
        _detail::remove(tree_node_type{std::dynamic_pointer_cast<NodeObject>(shared_from_this())}, path);
    }
}

//------------------------------------------------------------------

std::shared_ptr<NodeObject> NodeObject::create(const std::string& url)
{
    NodeObject* p = nullptr;
    if (url != "")
    {
        p = ::sp::utility::Factory<::sp::db::NodeObject>::create(url).release();
    }
    else
    {
        p = dynamic_cast<NodeObject*>(new NodeObjectDefault());
    }

    if (p == nullptr)
    {
        RUNTIME_ERROR << "Can not load plugin for url :" << url;
    }
    else
    {
        p->load(url);
    }

    return std::shared_ptr<NodeObject>(p);
}

} // namespace sp::db

namespace sp::utility
{

std::ostream& fancy_print(std::ostream& os, const sp::db::tree_node_type& v, int indent, int tab)
{
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<sp::db::tree_node_tags::Array, sp::db::tree_node_type>& array_p) {
                os << "[";

                array_p->for_each([&](const sp::db::Path::Segment&, sp::db::tree_node_type const& value) {
                    os << std::endl
                       << std::setw(indent * tab) << " ";
                    fancy_print(os, value, indent + 1, tab);
                    os << ",";
                });

                os << std::endl
                   << std::setw(indent * tab) << " "
                   << "]";
            },
            [&](const std::variant_alternative_t<sp::db::tree_node_tags::Object, sp::db::tree_node_type>& object_p) {
                os << "{";

                object_p->for_each(
                    [&](const sp::db::Path::Segment& key, sp::db::tree_node_type const& value) {
                        os << std::endl
                           << std::setw(indent * tab) << " "
                           << "\"" << std::get<std::string>(key) << "\" : ";
                        fancy_print(os, value, indent + 1, tab);
                        os << ",";
                    });

                os << std::endl
                   << std::setw(indent * tab) << " "
                   << "}";
            },
            //    [&](const std::variant_alternative_t<sp::db::tree_node_tags::Reference, sp::db::tree_node_type>& ref) {
            //        os << "<" << ref.second.str() << ">";
            //    },                                                                                                                                                                    //
            [&](const std::variant_alternative_t<sp::db::tree_node_tags::Block, sp::db::tree_node_type>& blk_p) { fancy_print(os, "<DATA BLOCK>", indent + 1, tab); }, //
            [&](const std::variant_alternative_t<sp::db::tree_node_tags::Null, sp::db::tree_node_type>& ele) { fancy_print(os, nullptr, indent + 1, tab); },           //
            [&](auto const& ele) { fancy_print(os, ele, indent + 1, tab); }                                                                                                       //
        },
        v);

    // if (entry.type() == Entry::NodeType::Element)
    // {
    //     os << to_string(entry.get_element());
    // }
    // else if (entry.type() == Entry::NodeType::Array)
    // else if (entry.type() == Entry::NodeType::Object)
    // {
    //
    // }
    return os;
}

} // namespace sp::utility