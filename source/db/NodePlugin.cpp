#include "NodePlugin.h"
#include "../utility/Logger.h"

namespace sp::db
{
typedef NodePlugin<std::map<std::string, Node>> NodeObjectDefault;

std::shared_ptr<NodeObject> NodeObject::create(const Node& opt)
{
    return std::dynamic_pointer_cast<NodeObject>(std::make_shared<NodeObjectDefault>(opt));
}

template <>
NodeObjectDefault::NodePlugin(const Node& opt) {}

template <>
void NodeObjectDefault::init(const std::initializer_list<Node>& init)
{
    for (auto& item : init)
    {
        auto& array = *std::get<Node::tags::Array>(item.value());

        Node(array.at(1)).swap(m_container_[array.at(0).as<Node::tags::String>()]);
    }
}

template <>
Cursor<Node> NodeObjectDefault::children()
{
    NOT_IMPLEMENTED;
    return Cursor<Node>{};
}

template <>
Cursor<const Node> NodeObjectDefault::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Node>{};
}

// template <>
// size_t NodeObjectDefault::size() const { return m_container_.size(); }

template <>
bool NodeObjectDefault::empty() const { return m_container_.size() == 0; }

// void NodeObjectDefault::for_each(std::function<void(const std::string&, Node&)> const& visitor)
// {
//     for (auto&& item : m_container_)
//     {
//         visitor(item.first, item.second);
//     }
// }
template <>
void NodeObjectDefault::for_each(std::function<void(const std::string&, const Node&)> const& visitor) const
{
    for (auto&& item : m_container_)
    {
        visitor(item.first, item.second);
    }
}

// template <>
// bool NodeObjectDefault::contain(const std::string& name) const { return m_container_.find(name) != m_container_.end(); }

// template <>
// void NodeObjectDefault::update_value(const std::string& name, Node&& v) { m_container_[name].swap(v); }

// template <>
// Node NodeObjectDefault::insert_value(const std::string& name, Node&& v)
// {
//     return Node{m_container_.emplace(name, std::move(v)).first->second};
// }

// template <>
// Node NodeObjectDefault::find_value(const std::string& name) const
// {
//     auto it = m_container_.find(name);
//     return it == m_container_.end() ? Node{} : Node{it->second};
// }

static std::map<std::string, std::function<Node(const Node&, const Node&)>> fetch_ops_map{

    {"$count", [](const Node& node, const Node& opt) {
         size_t res = 0;
         std::visit(
             sp::traits::overloaded{
                 [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                     res = std::dynamic_pointer_cast<NodeObjectDefault>(object_p)->container().size();
                 },
                 [&](const std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) { res = array_p->size(); },
                 [&](auto&& v) { res = 0; }},
             node.value());
         return Node(std::in_place_index_t<Node::tags::Int>(), res);
     }},
    {"$type", [](const Node& node, const Node& opt) { return Node(std::in_place_index_t<Node::tags::Int>(), node.type()); }},
};

Node fetch_op(const Node& node, const std::string& op, const Node& opt)
{
    Node res;
    auto it = fetch_ops_map.find(op);
    if (it != fetch_ops_map.end())
    {
        it->second(node, opt).swap(res);
    }
    else if (node.type() == Node::tags::Object)
    {
        Node(dynamic_cast<const NodeObjectDefault&>(node.as_object()).container().at(op)).swap(res);
    }

    return std::move(res);
}

static std::map<std::string, std::function<Node(Node&, const Node&)>> update_ops_map{

    {"$resize", [](Node& node, const Node& opt) {
         node.as_array().resize(opt.as<int>());
         return Node{};
     }},
    // {"$set", [](Node& node, const Node& opt) { return Node(std::in_place_index_t<Node::tags::Int>(), node.type()); }},

}; // namespace sp::db

Node update_op(Node& node, const std::string& key, const Node& opt)
{
    Node res;
    auto it = update_ops_map.find(key);
    if (it != update_ops_map.end())
    {
        it->second(node, opt).swap(res);
    }
    else if (node.type() == Node::tags::Object)
    {
        Node(opt).swap(dynamic_cast<NodeObjectDefault&>(node.as_object()).container()[key]);
    }

    return std::move(res);
}

template <>
Node NodeObjectDefault::update(const Path& path, const Node& ops, const Node& opt)
{

    Node root(std::in_place_index_t<Node::tags::Object>(), this->shared_from_this());

    Node* self = &root;

    for (auto it = path.begin(), ie = path.end(); it != ie; ++it)
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<Node::tags::Null, Node::value_type>&) {
                    self = &dynamic_cast<NodeObjectDefault&>(self->as_object()).container()[std::get<Path::segment_tags::Key>(*it)];
                },
                [&](std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                    self = &std::dynamic_pointer_cast<NodeObjectDefault>(object_p)->container()[std::get<Path::segment_tags::Key>(*it)];
                },
                [&](std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) {
                    self = &array_p->at(std::get<Path::segment_tags::Index>(*it));
                },
                [&](std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) { NOT_IMPLEMENTED; },
                [&](auto&& v) { NOT_IMPLEMENTED; }},
            self->value());
    }

    Node res;

    if (self == nullptr)
    {
        NOT_IMPLEMENTED;
    }
    else if (ops.type() == Node::tags::Object)
    {
        auto tmp = std::make_shared<NodeObjectDefault>();

        ops.as_object().for_each([&](const std::string& key, const Node& d) { tmp->m_container_.emplace(key, update_op(*self, key, d)); });

        if (tmp->container().size() == 1)
        {
            Node(tmp->container().begin()->second).swap(res);
        }
        else
        {
            Node(std::in_place_index_t<Node::tags::Object>(), tmp).swap(res);
        }
    }
    return std::move(res);
}

template <>
Node NodeObjectDefault::fetch(const Path& path, const Node& ops, const Node& opt) const
{
    Node root(const_cast<NodeObjectDefault*>(this)->shared_from_this());

    const Node* self = &root;

    for (auto it = path.begin(), ie = path.end(); it != ie; ++it)
    {
        std::visit(
            sp::traits::overloaded{
                [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                    self = &std::dynamic_pointer_cast<NodeObjectDefault>(object_p)->container().at(std::get<Path::segment_tags::Key>(*it));
                },
                [&](const std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) {
                    self = &array_p->at(std::get<Path::segment_tags::Index>(*it));
                },
                [&](const std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) {
                    NOT_IMPLEMENTED;
                    self = nullptr;
                },
                [&](auto&& v) { self = nullptr; }},
            self->value());
    }

    Node res;

    if (self != nullptr)
    {
        NOT_IMPLEMENTED;
    }
    else if (ops.type() == Node::tags::Object)
    {
        auto tmp = std::make_shared<NodeObjectDefault>();

        ops.as_object().for_each([&](const std::string& key, const Node& d) { tmp->m_container_.emplace(key, fetch_op(*self, key, d)); });

        if (tmp->container().size() == 1)
        {
            Node(tmp->container().begin()->second).swap(res);
        }
        else
        {
            Node(std::in_place_index_t<Node::tags::Object>(), tmp).swap(res);
        }
    }

    return std::move(res);
}
} // namespace sp::db
// NodeObject NodeObject::create(const Node & opt)
// {
//     // VERBOSE << "Load plugin for url:" << opt;
//     // NodeObject* p = nullptr;
//     // if (opt.index() == Node::tags::String)
//     // {
//     //     p = ::sp::utility::Factory<::sp::db::NodeObject>::create(std::get<Node::tags::String>(opt)).release();
//     // }
//     // else if (opt.index() == Node::tags::Null)
//     // {
//     //     p = new NodeObjectDefault();
//     // }

//     // if (p == nullptr)
//     // {
//     //     RUNTIME_ERROR << "Can not load plugin for url :" << opt;
//     // }
//     // else
//     // {
//     //     p->load(opt);
//     // }

//     // return std::shared_ptr<NodeObject>(p);
// }

// namespace _detail
// {

// Node insert(Node self, Path::Segment path_seg, Node v)
// {

//     std::visit(
//         sp::traits::overloaded{
//             [&](std::variant_alternative_t<Node::tags::Object, Node>& object_p) {
//                 object_p->insert(path_seg, Node{std::in_place_index_t<Node::tags::Object>()}).swap(self);
//             },
//             [&](std::variant_alternative_t<Node::tags::Array, Node>& array_p) {
//                 array_p->insert(std::get<Path::segment_tags::Index>(path_seg), Node{std::in_place_index_t<Node::tags::Object>()}).swap(self);
//             },
//             [&](std::variant_alternative_t<Node::tags::Block, Node>&) {
//                 NOT_IMPLEMENTED;
//             },
//             [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
//         self);

//     return std::move(self);
// }

// Node insert(Node self, Path path, Node v)
// {
//     for (auto it = path.begin(), ie = --path.end(); it != ie; ++it)
//     {
//         insert(self, *it, Node{std::in_place_index_t<Node::tags::Object>()});
//     }
//     insert(self, path.last(), std::move(v)).swap(self);
//     return self;
// }
// void update(Node self, Path path, Node v)
// {
//     for (auto it = path.begin(), ie = --path.end(); it != ie; ++it)
//     {
//         insert(self, *it, Node{std::in_place_index_t<Node::tags::Object>()}).swap(self);
//     }

//     std::visit(
//         sp::traits::overloaded{
//             [&](std::variant_alternative_t<Node::tags::Object, Node>& object_p) {
//                 object_p->update(path.last(), std::move(v));
//             },
//             [&](std::variant_alternative_t<Node::tags::Array, Node>& array_p) {
//                 array_p->insert(std::get<Path::segment_tags::Index>(path.last()), std::move(v));
//             },
//             [&](std::variant_alternative_t<Node::tags::Block, Node>&) {
//                 NOT_IMPLEMENTED;
//             },
//             [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
//         self);
// }

// Node find(Node self, Path path)
// {
//     bool found = true;
//     Path prefix;

//     for (auto it = path.begin(), ie = path.end(); it != ie; ++it)
//     {

//         prefix.append(*it);

//         switch (self.index())
//         {
//         case Node::tags::Object:
//             Node(std::get<Node::tags::Object>(self)->find(*it)).swap(self);
//             break;
//         case Node::tags::Array:
//             Node(std::get<Node::tags::Array>(self)->at(std::get<Path::segment_tags::Index>(*it))).swap(self);
//             break;
//         default:
//             found = false;
//             break;
//         }
//     }
//     if (!found)
//     {
//         throw std::out_of_range(FILE_LINE_STAMP_STRING + "Can not find url:" + prefix.str());
//     }

//     return std::move(self);
// }

// void remove(Node self, Path path)
// {
//     find(self, path.prefix()).swap(self);

//     std::visit(
//         sp::traits::overloaded{
//             [&](std::variant_alternative_t<Node::tags::Object, Node>& object_p) {
//                 object_p->remove(path.last());
//             },
//             [&](std::variant_alternative_t<Node::tags::Array, Node>& array_p) {
//                 array_p->at(std::get<Path::segment_tags::Index>(path.last())).emplace<Node::tags::Null>(nullptr);
//             },
//             [&](std::variant_alternative_t<Node::tags::Block, Node>&) {
//                 NOT_IMPLEMENTED;
//             },
//             [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
//         self);
// }

// } // namespace _detail
