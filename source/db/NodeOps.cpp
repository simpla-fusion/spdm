#include "NodeOps.h"
#include "../utility/Logger.h"
#include "../utility/TypeTraits.h"
#include "Node.h"
#include <variant>

namespace sp::db
{

static std::map<std::string, std::function<Node(const Node&, const Node&)>> fetch_ops_map{

    {"$size", [](const Node& node, const Node& opt) {
         size_t res = 0;
         std::visit(
             sp::traits::overloaded{
                 [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) { res = object_p.size(); },
                 [&](const std::variant_alternative_t<Node::tags::Array, Node::value_type>& array_p) { res = array_p.size(); },
                 [&](auto&& v) { res = 0; }},
             node.get_value());
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
        node.as_object().find_value(op).swap(res);
    }

    return std::move(res);
}
} // namespace sp::db