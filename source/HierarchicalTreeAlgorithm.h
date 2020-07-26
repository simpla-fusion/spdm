
#ifndef SP_HIERACHICAL_TREE_ALGORITHM_H_
#define SP_HIERACHICAL_TREE_ALGORITHM_H_
#include "HierarchicalTree.h"
#include "utility/Logger.h"
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
template <template <typename> class TmplObject,
          template <typename> class TmplArray,
          typename... TypeList>
class HierarchicalTreeTmpl;

template <typename TreeType>
class Cursor;

template <template <typename> class TmplObject,
          template <typename> class TmplArray,
          typename... TypeList>
class Cursor<HierarchicalTreeTmpl<TmplObject, TmplArray, TypeList...>>
{

public:
    typedef HierarchicalTreeTmpl<TmplObject, TmplArray, TypeList...> tree_type;

    cursor();

    cursor(tree_type*);

    cursor(const std::shared_ptr<tree_type>&);

    cursor(const cursor&);

    cursor(cursor&&);

    ~cursor() = default;

    bool operator==(const cursor& other) const;

    bool operator!=(const cursor& other) const;

    operator bool() const { return !is_null(); }

    bool is_null() const;

    tree_type& operator*();

    tree_type* operator->();

    cursor& operator++() { return next(); }

    cursor operator++(int);

    cursor next() const;
    cursor& next();

private:
    std::shared_ptr<tree_type> m_node_;
};

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