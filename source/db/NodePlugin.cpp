#include "NodePlugin.h"
#include "../utility/Logger.h"
#include "NodeOps.h"
namespace sp::db
{
typedef NodePlugin<std::map<std::string, Node>> NodeBackendDefault;

std::shared_ptr<NodeBackend> NodeBackend::create(const NodeObject& opt)
{
    return std::dynamic_pointer_cast<NodeBackend>(std::make_shared<NodeBackendDefault>(opt));
}

template <>
NodeBackendDefault::NodePlugin(const NodeObject& opt) {}

template <>
Cursor<Node> NodeBackendDefault::children()
{
    NOT_IMPLEMENTED;
    return Cursor<Node>{};
}

template <>
Cursor<const Node> NodeBackendDefault::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Node>{};
}

template <>
size_t NodeBackendDefault::size() const { return m_container_.size(); }

// void NodeBackendDefault::for_each(std::function<void(const std::string&, Node&)> const& visitor)
// {
//     for (auto&& item : m_container_)
//     {
//         visitor(item.first, item.second);
//     }
// }
template <>
void NodeBackendDefault::for_each(std::function<void(const std::string&, const Node&)> const& visitor) const
{
    for (auto&& item : m_container_)
    {
        visitor(item.first, item.second);
    }
}

template <>
bool NodeBackendDefault::contain(const std::string& name) const { return m_container_.find(name) != m_container_.end(); }

template <>
void NodeBackendDefault::update_value(const std::string& name, Node&& v) { m_container_[name].swap(v); }

template <>
Node NodeBackendDefault::insert_value(const std::string& name, Node&& v)
{
    return Node{m_container_.emplace(name, std::move(v)).first->second};
}

template <>
Node NodeBackendDefault::find_value(const std::string& name) const
{
    auto it = m_container_.find(name);
    return it == m_container_.end() ? Node{} : Node{it->second};
}

template <>
void NodeBackendDefault::update(const Path& path, const Node& patch, const NodeObject& opt)
{

    Node self{std::in_place_index_t<Node::tags::Object>(), this->shared_from_this()};

    for (auto it = path.begin(), ie = --path.end(); it != ie && self.type() != Node::tags::Null; ++it)
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                    object_p.insert_value(std::get<Path::segment_tags::Key>(*it), NodeObject{std::make_shared<NodeBackendDefault>()}).swap(self);
                },
                [&](std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) {
                    array_p.insert(std::get<Path::segment_tags::Index>(*it), NodeObject{std::make_shared<NodeBackendDefault>()}).swap(self);
                },
                [&](std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) { NOT_IMPLEMENTED; },
                [&](auto&& v) { NOT_IMPLEMENTED; }},
            self.get_value());
    }

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                object_p.update_value(std::get<Path::segment_tags::Key>(path.last()), Node{patch});
            },
            [&](std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) {
                Node{patch}.swap(array_p.at(std::get<Path::segment_tags::Index>(path.last())));
            },
            [&](std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) { NOT_IMPLEMENTED; },
            [&](auto&& v) {  std::cerr<<path<<std::endl; 
            NOT_IMPLEMENTED; }},
        self.get_value());
}

template <>
Node NodeBackendDefault::merge(const Path& path, const Node& patch, const NodeObject& opt)
{
    Node self{std::in_place_index_t<Node::tags::Object>(), this->shared_from_this()};

    for (auto it = path.begin(), ie = --path.end(); it != ie && self.type() != Node::tags::Null; ++it)
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                    object_p.insert_value(std::get<Path::segment_tags::Key>(*it), NodeObject{std::make_shared<NodeBackendDefault>()}).swap(self);
                },
                [&](std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) {
                    array_p.insert(std::get<Path::segment_tags::Index>(*it), NodeObject{std::make_shared<NodeBackendDefault>()}).swap(self);
                },
                [&](std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) { NOT_IMPLEMENTED; },
                [&](auto&& v) { NOT_IMPLEMENTED; }},
            self.get_value());
    }

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                Node(object_p.insert_value(std::get<Path::segment_tags::Key>(path.last()), Node{patch})).swap(self);
            },
            [&](std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) {
                Node(array_p.insert(std::get<Path::segment_tags::Index>(path.last()), Node{patch})).swap(self);
            },
            [&](std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) { NOT_IMPLEMENTED; },
            [&](auto&& v) { NOT_IMPLEMENTED; }},
        self.get_value());

    return std::move(self);
}

template <>
Node NodeBackendDefault::fetch(const Path& path, const Node& projection, const NodeObject& opt) const
{
    Node self{std::in_place_index_t<Node::tags::Object>(), const_cast<NodeBackendDefault*>(this)->shared_from_this()};

    for (auto it = path.begin(), ie = path.end(); it != ie && self.type() != Node::tags::Null; ++it)
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) { object_p.find_value(std::get<Path::segment_tags::Key>(*it)).swap(self); },
                [&](std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) { array_p.at(std::get<Path::segment_tags::Index>(*it)).swap(self); },
                [&](std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) { self.as<Node::tags::Block>(blk.slice(std::get<Path::segment_tags::Slice>(*it))); },
                [&](auto&& v) { self.as<Node::tags::Null>(nullptr); }},
            self.get_value());
    }

    if (self.type() == Node::tags::Null)
    {
        NOT_IMPLEMENTED;
    }
    else if (projection.type() == Node::tags::Object)
    {
        auto res = std::make_shared<NodeBackendDefault>();

        projection.as_object().for_each([&](const std::string& key, const Node& d) {
            res->container().emplace(key, fetch_op(self, key, d));
        });

        if (projection.as_object().size() == 1)
        {
            Node(res->container().begin()->second).swap(self);
        }
        else
        {
            Node(std::in_place_index_t<Node::tags::Object>(), res).swap(self);
        }
    }

    return std::move(self);
}
} // namespace sp::db
// NodeObject NodeObject::create(const NodeObject& opt)
// {
//     // VERBOSE << "Load plugin for url:" << opt;
//     // NodeObject* p = nullptr;
//     // if (opt.index() == Node::tags::String)
//     // {
//     //     p = ::sp::utility::Factory<::sp::db::NodeObject>::create(std::get<Node::tags::String>(opt)).release();
//     // }
//     // else if (opt.index() == Node::tags::Null)
//     // {
//     //     p = new NodeBackendDefault();
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
