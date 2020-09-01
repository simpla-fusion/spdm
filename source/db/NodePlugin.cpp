#include "NodePlugin.h"
#include "../utility/Logger.h"
namespace sp::db
{

//----------------------------------------------------------------------------------------------------
typedef NodePlugin<std::map<std::string, Node>> NodeBackendDefault;

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

template <>
bool NodeBackendDefault::contain(const std::string& name) const { return m_container_.find(name) != m_container_.end(); }

template <>
void NodeBackendDefault::update_value(const std::string& name, Node&& v) { m_container_[name].swap(v); }

template <>
Node NodeBackendDefault::insert_value(const std::string& name, Node&& v) { return m_container_.emplace(name, v).first->second; }

template <>
Node NodeBackendDefault::find_value(const std::string& name) const
{
    auto it = m_container_.find(name);
    return it == m_container_.end() ? Node{} : it->second;
}

template <>
void NodeBackendDefault::update(const Path& path, const Node& patch, const NodeObject& opt) { NOT_IMPLEMENTED; }

template <>
Node NodeBackendDefault::merge(const Path& path, const Node& patch, const NodeObject& opt)
{
    NOT_IMPLEMENTED;
    return Node{};
}

template <>
Node NodeBackendDefault::fetch(const Path&, const Node& projection, const NodeObject& opt) const
{

    NOT_IMPLEMENTED;

    switch (projection.type())
    {
    case Node::tags::Object:
        // node.as<Node::tags::Object>().for_each();

    default:
        break;
    }
    // switch (path.length())
    // {
    // case 0:
    //     NOT_IMPLEMENTED;
    //     break;
    // case 1:
    //     Node(m_container_.emplace(std::get<Path::segment_tags::Key>(*path.begin()), std::move(v)).first->second).swap(res);
    //     break;
    // default:
    //     _detail::insert(Node{std::dynamic_pointer_cast<NodeObject>(shared_from_this())}, path, std::move(v)).swap(res);
    //     break;
    // }

    // switch (path.length())
    // {
    // case 0:
    //     NOT_IMPLEMENTED;
    //     break;
    // case 1:
    //     m_container_[std::get<Path::segment_tags::Key>(*path.begin())].swap(v);
    //     break;
    // default:
    //     _detail::update(Node{std::dynamic_pointer_cast<NodeObject>(shared_from_this())}, path, std::move(v));
    // }

    // Node self;
    // switch (path.length())
    // {
    // case 0:
    //     m_container_.clear();
    //     break;
    // case 1:
    //     m_container_.erase(std::get<Path::segment_tags::Key>(*path.begin()));
    //     break;
    // default:
    //     _detail::remove(Node{std::dynamic_pointer_cast<NodeObject>(shared_from_this())}, path);
    // }
    return Node{};
}

//------------------------------------------------------------------

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

std::shared_ptr<NodeBackend> NodeBackend::create(const NodeObject& opt)
{
    return std::dynamic_pointer_cast<NodeBackend>(std::make_shared<NodeBackendDefault>(opt));
}

} // namespace sp::db