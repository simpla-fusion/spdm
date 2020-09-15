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
bool NodeObjectDefault::empty() const { return m_container_.size() == 0; }

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

static std::map<std::string, std::function<Node(Node&, const Node&)>> update_ops_map{

    {"$resize", [](Node& node, const Node& opt) {
         node.as_array().resize(opt.get_value<int>());
         return Node{};
     }},
    {"$set", [](Node& node, const Node& opt) {
         Node(opt).swap(node);
         return Node(node);
     }},
    {"$default", [](Node& node, const Node& opt) {
         if (node.type() == Node::tags::Null)
         {
             Node(opt).swap(node);
         }
         return Node(node);
     }},
    {"$push_back", [](Node& node, const Node& opt) {
         node.as_array().push_back(opt);
         return Node(static_cast<int>(node.as_array().size() - 1));
     }},
    {"$pop_back", [](Node& node, const Node& opt) {
         node.as_array().pop_back();
         return Node(static_cast<int>(node.as_array().size() - 1));
     }},

};

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
void NodeObjectDefault::update(const Node& query, const Node& ops)
{

    Node root(std::in_place_index_t<Node::tags::Object>(), this->shared_from_this());

    Node* self = &root;

    if (query.type() != Node::tags::Path)
    {
        NOT_IMPLEMENTED;
    }
    else
    {
        auto path = query.as<Node::tags::Path>();

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

        if (self == nullptr)
        {
            RUNTIME_ERROR << "Illegal path! " << path.str();
            // throw std::runtime_error("Illegal path! " + path.str());
        }
    }

    if (ops.type() == Node::tags::Object)
    {
        auto tmp = std::make_shared<NodeObjectDefault>();

        ops.as_object().for_each([&](const Node& key, const Node& d) {
            tmp->m_container_.emplace(key.get_value<Node::tags::String>(), update_op(*self, key.get_value<Node::tags::String>(), d));
        });

        // if (tmp->container().size() == 1)
        // {
        //     Node(tmp->container().begin()->second).swap(res);
        // }
        // else
        // {
        //     Node(std::in_place_index_t<Node::tags::Object>(), tmp).swap(res);
        // }
    }
    else
    {
        Node(ops).swap(*self);
    }
}

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
         return Node(std::in_place_index_t<Node::tags::Integer>(), res);
     }},
    {"$type", [](const Node& node, const Node& opt) { return Node(std::in_place_index_t<Node::tags::Integer>(), node.type()); }},
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

template <>
Node NodeObjectDefault::fetch(const Node& query, const Node& ops)
{
    NOT_IMPLEMENTED;
    return Node{};
}

template <>
const Node NodeObjectDefault::fetch(const Node& query, const Node& ops) const
{
    Node root(const_cast<NodeObjectDefault*>(this)->shared_from_this());

    Node* self = &root;

    if (query.type() == Node::tags::Path)
    {
        auto path = query.as<Node::tags::Path>();

        for (auto it = path.begin(), ie = path.end(); it != ie && self != nullptr; ++it)
        {
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

        if (self == nullptr)
        {
            RUNTIME_ERROR << "Illegal path! " << path.str();
            throw std::runtime_error("Illegal path! " + path.str());
        }
    }
    else if (query.type() == Node::tags::String)
    {
        auto key = self->as<Node::tags::String>();

        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                    auto& obj = std::dynamic_pointer_cast<NodeObjectDefault>(object_p)->container();
                    auto it = obj.find(key);
                    self = it == obj.end() ? nullptr : &it->second;
                },

                [&](auto&& v) { self = nullptr; }},
            self->value());
    }
    else
    {
        self = nullptr;
    }

    Node res;
    if (self == nullptr)
    {
    }
    else if (ops.type() == Node::tags::Object)
    {
        auto tmp = std::make_shared<NodeObjectDefault>();

        ops.as_object().for_each([&](const Node& key, const Node& d) {
            tmp->m_container_.emplace(
                key.get_value<std::string>(),
                fetch_op(*self, key.get_value<std::string>(), d));
        });

        if (tmp->container().size() == 1)
        {
            Node(tmp->container().begin()->second).swap(res);
        }
        else
        {
            Node(std::in_place_index_t<Node::tags::Object>(), tmp).swap(res);
        }
    }
    else
    {
        Node(*self).swap(res);
    }

    return std::move(res);
}

} // namespace sp::db
