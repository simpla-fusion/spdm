
#ifndef SP_HIERACHICAL_TREE_ALGORITHM_H_
#define SP_HIERACHICAL_TREE_ALGORITHM_H_
#include "HierarchicalTree.h"
#include "utility/Logger.h"
#include "utility/fancy_print.h"
#include <any>
#include <array>
#include <complex>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace sp
{

template <typename TNode, template <typename> class ObjectHolder>
std::ostream& fancy_print(std::ostream& os, const HierarchicalTreeObject<TNode, ObjectHolder>& tree_object, int indent, int tab)
{
    return fancy_print(os, tree_object.data(), indent, tab);
}
template <typename TNode, template <typename> class ArrayHolder>
std::ostream& fancy_print(std::ostream& os, const HierarchicalTreeArray<TNode, ArrayHolder>& tree_array, int indent, int tab)
{
    return fancy_print(os, tree_array.data(), indent, tab);
}

template <typename TNode,
          template <typename> class ObjectHolder,
          template <typename> class ArrayHolder,
          typename... TypeList>
std::ostream& fancy_print(std::ostream& os, const HierarchicalTree<TNode, ObjectHolder, ArrayHolder, TypeList...>& tree, int indent, int tab)
{
    std::visit([&](auto&& v) { fancy_print(os, v, indent, tab); }, tree.data());

    return os;
}

// typedef HierarchicalTree<TNode, ObjectHolder, ArrayHolder, TypeList...> node_type;

// if (tree.type() > node_type::ARRAY_TAG)
// {
// }
// else if (tree.type() == node_type::ARRAY_TAG)
// {
//     // os << "[";
//     // for (auto it = tree.first_child(); !it.is_null(); ++it)
//     // {
//     //     os << std::endl
//     //        << std::setw(indent * tab) << " ";
//     //     fancy_print(os, it->value(), indent + 1, tab);
//     //     os << ",";
//     // }
//     // os << std::endl
//     //    << std::setw(indent * tab)
//     //    << "]";
// }
// else if (tree.type() == node_type::OBJECT_TAG)
// {
//     // os << "{";
//     // for (auto it = tree.first_child(); !it.is_null(); ++it)
//     // {
//     //     os << std::endl
//     //        << std::setw(indent * tab) << " "
//     //        << "\"" << it->name() << "\" : ";
//     //     fancy_print(os, it->value(), indent + 1, tab);
//     //     os << ",";
//     // }
//     // os << std::endl
//     //    << std::setw(indent * tab)
//     //    << "}";
// }
// return os;
template <typename TNode,
          template <typename> class ObjectHolder,
          template <typename> class ArrayHolder,
          typename... TypeList>
std::ostream& operator<<(std::ostream& os, const HierarchicalTree<TNode, ObjectHolder, ArrayHolder, TypeList...>& tree)
{
    return fancy_print(os, tree, 0, 4);
}

// size_t depth() ; // parent.depth +1

// size_t height() ; // max(children.height) +1

// cursor slibings() ; // return slibings

// cursor ancestor() ; // return ancestor

// cursor descendants() ; // return descendants

// cursor leaves() ; // return leave nodes in traversal order

// cursor shortest_path(Node & target) ; // return the shortest path to target

// ptrdiff_t distance(const this_type& target) ; // lenght of shortest path to target
} // namespace sp

#endif //SP_HIERACHICAL_TREE_ALGORITHM_H_