#include "NodePlugin.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <cassert>

namespace sp::db
{
typedef NodePlugin<std::map<std::string, Node>> NodeObjectDefault;

//==========================================================================================
// NodeObject

std::shared_ptr<NodeObject> create_node_object(const Node& opt)
{
    std::shared_ptr<NodeObject> res = nullptr;

    opt.visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<sp::db::Node::tags::String, sp::db::Node::value_type>& uri) {
                res = std::shared_ptr<NodeObject>(sp::utility::Factory<NodeObject>::create(uri).release());
                if (res != nullptr)
                {
                    res->load(opt);
                }
            },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Object, sp::db::Node::value_type>& object_p) {
                auto schema = object_p->find_child(DEFAULT_SCHEMA_TAG).get_value<Node::tags::String>("");
                if (schema == "")
                {
                    res = object_p;
                }
                else
                {
                    res = std::shared_ptr<NodeObject>(sp::utility::Factory<NodeObject>::create(schema).release());
                    res->load(opt);
                }
            },
            [&](auto&& ele) {} //
        });

    if (res == nullptr)
    {
        res = std::make_shared<NodeObjectDefault>(opt);
        res->load(opt);
    }

    return res;
}

template <>
NodeObjectDefault::NodePlugin(const std::initializer_list<Node>& init) : m_container_()
{
    for (auto& item : init)
    {
        auto& array = *item.as<Node::tags::Array>();

        Node(array.at(1)).swap(m_container_[array.at(0).get_value<Node::tags::String>()]);
    }
}

template <>
void NodeObjectDefault::load(const Node& opt)
{
    std::visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<sp::db::Node::tags::Object, sp::db::Node::value_type>& object_p) {
                object_p->for_each([&](const Node& k, const Node& n) {
                    Node(n).swap(m_container_[k.get_value<std::string>()]);
                });
            },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Array, sp::db::Node::value_type>& array_p) {
                // NOT_IMPLEMENTED;
            },
            [&](auto&& ele) {} //
        },
        opt.value());
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

template <>
size_t NodeObjectDefault::count() const { return m_container_.size(); }

template <>
bool NodeObjectDefault::empty() const { return count() == 0; }

template <>
void NodeObjectDefault::for_each(const std::function<void(const Node&, Node&)>& visitor)
{
    for (auto&& item : m_container_)
    {
        visitor(item.first, item.second);
    }
}
template <>
bool NodeObjectDefault::contain(const std::string& key) const { return m_container_.find(key) != m_container_.end(); }

template <>
void NodeObjectDefault::update_child(const std::string& key, const Node& node) { Node(node).swap(m_container_[key]); }

template <>
Node NodeObjectDefault::insert_child(const std::string& key, const Node& node) { return m_container_.emplace(key, node).first->second; }

template <>
Node NodeObjectDefault::find_child(const std::string& key) const
{
    auto it = m_container_.find(key);
    return it == m_container_.end() ? Node{} : Node(it->second);
}
template <>
void NodeObjectDefault::remove_child(const std::string& key)
{
    auto it = m_container_.find(key);
    if (it != m_container_.end())
    {
        m_container_.erase(it);
    };
}

template <>
void NodeObjectDefault::for_each(const std::function<void(const Node&, const Node&)>& visitor) const
{
    for (auto&& item : m_container_)
    {
        visitor(item.first, item.second);
    }
}

template <>
Node NodeObjectDefault::update(const Path& path, int op, const Node& data)
{
    Node root(std::in_place_index_t<Node::tags::Object>(), this->shared_from_this());

    Node* self = &root;

    for (auto it = path.begin(), ie = path.end(); it != ie; ++it)
    {
        if (it->index() > Path::tags::Index)
        {
            NOT_IMPLEMENTED;
        }
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<Node::tags::Null, Node::value_type>&) {
                    auto obj = std::make_shared<NodeObjectDefault>();
                    self->value().emplace<Node::tags::Object>(obj);
                    self = &(obj->container()[std::get<Path::tags::Key>(*it)]);
                },
                [&](std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                    self = &(std::dynamic_pointer_cast<NodeObjectDefault>(object_p)->container()[std::get<Path::tags::Key>(*it)]);
                },
                [&](std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) {
                    self = &(array_p->at(std::get<Path::tags::Index>(*it)));
                },
                [&](std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) { NOT_IMPLEMENTED; },
                [&](auto&& v) { NOT_IMPLEMENTED; }},
            self->value());
    }

    if (self == nullptr) THROW_EXCEPTION_OUT_OF_RANGE("Illegal path! ", path);

    Node res;

    switch (op)
    {
    case Node::ops::SET:
        Node(data).swap(*self);
        break;
    case Node::ops::RESIZE:
        self->as_array().resize(data.get_value<Node::tags::Integer>());
        break;
    case Node::ops::PUSH_BACK:
        self->as_array().push_back(data);
        res.set_value<Node::tags::Integer>(self->as_array().count() - 1);
        break;
    case Node::ops::POP_BACK:
        self->as_array().pop_back().swap(res);
        break;
    default:
        break;
    }

    return std::move(res);
}

template <>
const Node NodeObjectDefault::fetch(const Path& path, int op, const Node& data) const
{
    Node root(const_cast<NodeObjectDefault*>(this)->shared_from_this());

    Node* self = &root;

    Path prefix;

    for (auto it = path.begin(), ie = path.end(); it != ie; ++it)
    {

        if (self == nullptr) THROW_EXCEPTION_OUT_OF_RANGE("Illegal path! ", prefix);

        prefix.append(*it);

        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                    auto& obj = std::dynamic_pointer_cast<NodeObjectDefault>(object_p)->container();
                    auto k_it = obj.find(std::get<Path::tags::Key>(*it));
                    self = k_it == obj.end() ? nullptr : &k_it->second;
                },
                [&](std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) {
                    self = &array_p->at(std::get<Path::tags::Index>(*it));
                },
                [&](std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) {
                    NOT_IMPLEMENTED;
                    self = nullptr;
                },
                [&](auto&& v) { self = nullptr; }},
            self->value());
    }

    Node res;

    switch (op)
    {
    case Node::ops::TYPE:
        res.set_value<Node::tags::Integer>(self == nullptr ? Node::tags::Null : self->type());
        break;
    case Node::ops::COUNT:
        if (self == nullptr)
        {
            res.set_value<Node::tags::Integer>(0);
        }
        else
        {

            std::visit(
                traits::overloaded{
                    [&](const std::variant_alternative_t<Node::tags::Null, Node::value_type>& blk) {
                        res.set_value<Node::tags::Integer>(0);
                    },
                    [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                        res.set_value<Node::tags::Integer>(object_p->count());
                    },
                    [&](const std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) {
                        res.set_value<Node::tags::Integer>(array_p->count());
                    },
                    [&](const std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) {
                        NOT_IMPLEMENTED;
                        self = nullptr;
                    },
                    [&](auto&& v) {
                        res.set_value<Node::tags::Integer>(1);
                    }},
                const_cast<const Node*>(self)->value());
        }
        break;
    case Node::ops::GET:
        if (self == nullptr)
        {
            Node(data).swap(res);
        }
        else
        {
            Node(*self).swap(res);

            // self->visit(
            //     traits::overloaded{
            //         [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& p_obj) {},
            //         [&](auto&& v) { Node(*self).swap(res); }});

            // if (data.type() == Node::tags::Object)
            // {
            //     auto tmp = std::make_shared<NodeObjectDefault>();

            //     data.as_object().for_each([&](const Node& key, const Node& d) {
            //         tmp->m_container_.emplace(
            //             key.get_value<std::string>(),
            //             fetch_op(*self, key.get_value<std::string>(), d));
            //     });

            //     if (tmp->container().size() == 1)
            //     {
            //         Node(tmp->container().begin()->second).swap(res);
            //     }
            //     else
            //     {
            //         Node(std::in_place_index_t<Node::tags::Object>(), tmp).swap(res);
            //     }
            // }
            // else
            // {
            //     Node(*self).swap(res);
            // }
        }
        break;

    default:
        break;
    }

    return std::move(res);
}

} // namespace sp::db
