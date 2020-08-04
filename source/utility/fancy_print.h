/**
 * @file  fancy_print.h
 *
 *  created on: 2013-11-29
 *      Author: salmon
 *  * change (20200713 salmon): 
 *      - filename => fancy_print.h
 *      - function name  => fancy_print_XXXX
 */

#ifndef SP_FANCY_PRINT_H_
#define SP_FANCY_PRINT_H_

#include <array>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
namespace sp::utility
{
#define ENABLE_IF(_COND_) std::enable_if_t<_COND_, void>* _p = nullptr

/**
     * @ingroup toolbox
     * @addtogroup fancy_print Fancy print
     * @{
     */

template <typename TV, typename TI>
inline TV const* print_nd_array(std::ostream& os, TV const* v, int rank, TI const* d, bool is_first = true,
                                bool is_last = true, std::string const& left_brace = "{", std::string const& sep = ",",
                                std::string const& right_brace = "}", bool is_slow_first = true, int tab_width = 0,
                                int indent = 0, int tab = 1)
{
    constexpr int ELE_NUM_PER_LINE = 10;
    if (rank == 1)
    {
        os << left_brace;
        for (int s = 0; s < d[0]; ++s)
        {
            if (s > 0)
            {
                os << sep;
            }
            if (s % ELE_NUM_PER_LINE == 0 && s != 0)
            {
                os << std::endl;
            }
            if (tab_width > 0)
            {
                os << std::setw(10) << " ";
            }
            os << (*v);
            ++v;
        }
        os << right_brace;
    }
    else
    {
        os << left_brace;

        for (int s = 0; s < d[0]; ++s)
        {
            if (s > 0)
            {
                os << sep;
                //                if (rank > 1) { os << std::endl << std::setw(tab_width) << " "; }
            }
            v = print_nd_array(os, v, rank - 1, d + 1, s == 0, s == d[0] - 1, left_brace, sep, right_brace, is_slow_first,
                               tab_width, indent + tab, tab);
        }
        os << right_brace;
        return (v);
    }
    //    if (is_last) { os << std::endl; }
    return v;
}

namespace detail
{
template <size_t... IDX, typename V, typename I>
auto get_value(std::integer_sequence<size_t, IDX...>, V const& v, I const* idx)
{
    return v(idx[IDX]...);
};
} // namespace detail

template <int NDIMS, typename TV, typename TI>
std::ostream& fancy_print_nd_slow_first(std::ostream& os, TV const& v, int depth, int* idx, TI const* lo,
                                        TI const* hi, int indent = 0, int tab_width = 0,
                                        std::string const& left_brace = "{", std::string const& sep = ",",
                                        std::string const& right_brace = "}")
{
    if (depth >= NDIMS)
    {
        return os;
    }

    if (depth == NDIMS - 1)
    {
        os << std::setw(indent) << left_brace;
        idx[NDIMS - 1] = lo[NDIMS - 1];
        os << std::setw(tab_width) << detail::get_value(std::make_index_sequence<NDIMS>(), v, idx);
        for (idx[NDIMS - 1] = lo[NDIMS - 1] + 1; idx[NDIMS - 1] < hi[NDIMS - 1]; ++idx[NDIMS - 1])
        {
            os << sep << std::setw(tab_width) << detail::get_value(std::make_index_sequence<NDIMS>(), v, idx);
        }
        os << right_brace;
    }
    else
    {
        os << std::setw(indent) << left_brace << std::endl;
        idx[depth] = lo[depth];
        fancy_print_nd_slow_first<NDIMS>(os, v, depth + 1, idx, lo, hi, indent + tab_width, tab_width, left_brace, sep,
                                         right_brace);

        for (idx[depth] = lo[depth] + 1; idx[depth] < hi[depth]; ++idx[depth])
        {
            os << sep << std::endl;
            fancy_print_nd_slow_first<NDIMS>(os, v, depth + 1, idx, lo, hi, indent + tab_width, tab_width, left_brace, sep,
                                             right_brace);
        }
        os << std::endl
           << std::setw(indent) << right_brace;
    }

    return os;
}

template <int NDIMS, typename TV, typename TI>
std::ostream& fancy_print_nd_fast_first(std::ostream& os, TV const& v, int depth, int* idx, TI const* lo,
                                        TI const* hi, int indent = 0, int tab_width = 0,
                                        std::string const& left_brace = "{", std::string const& sep = ",",
                                        std::string const& right_brace = "}")
{
    if (depth < 0)
    {
        return os;
    }

    if (depth == 0)
    {
        os << std::setw(indent + depth) << left_brace;
        idx[0] = lo[0];
        os << std::setw(tab_width) << detail::get_value(std::make_index_sequence<NDIMS>(), v, idx);
        for (idx[0] = lo[0] + 1; idx[0] < hi[0]; ++idx[0])
        {
            os << "," << std::setw(tab_width) << detail::get_value(std::make_index_sequence<NDIMS>(), v, idx);
        }
        os << right_brace;
    }
    else
    {
        os << std::setw(indent) << left_brace << std::endl;
        idx[depth] = lo[depth];
        fancy_print_nd_fast_first<NDIMS>(os, v, depth - 1, idx, lo, hi, indent + tab_width, tab_width, left_brace, sep,
                                         right_brace);

        for (idx[depth] = lo[depth] + 1; idx[depth] < hi[depth]; ++idx[depth])
        {
            os << "," << std::endl;
            fancy_print_nd_fast_first<NDIMS>(os, v, depth - 1, idx, lo, hi, indent + tab_width, tab_width, left_brace, sep,
                                             right_brace);
        }
        os << std::endl
           << std::setw(indent) << right_brace;
    }

    return os;
}

template <int NDIMS, typename TV, typename TI>
std::ostream& fancy_print_nd(std::ostream& os, TV const& v, TI const* lo, TI const* hi, bool is_slow_first = true,
                             int indent = 0, int tab = 1, int tab_width = 8, std::string const& left_brace = "{",
                             std::string const& sep = ",", std::string const& right_brace = "}")
{
    constexpr int ELE_NUM_PER_LINE = 10;

    int idx[NDIMS];
    if (is_slow_first)
    {
        fancy_print_nd_slow_first<NDIMS>(os, v, 0, idx, lo, hi, indent, tab_width, left_brace, sep, right_brace);
    }
    else
    {
        fancy_print_nd_fast_first<NDIMS>(os, v, NDIMS - 1, idx, lo, hi, indent, tab_width, left_brace, sep, right_brace);
    }

    return os;
}
template <typename TX, typename TY, typename... Others>
std::istream& get_(std::istream& is, size_t num, std::map<TX, TY, Others...>& a)
{
    for (size_t s = 0; s < num; ++s)
    {
        TX x;
        TY y;
        is >> x >> y;
        a.emplace(x, y);
    }
    return is;
}
template <typename V>
std::ostream& fancy_print(std::ostream& os, V const* d, size_t ndims, size_t const* extents, int indent, int tab)
{
    print_nd_array(os, d, ndims, extents, true, false, "[", ",", "]", true, 0, indent);
    return os;
}

template <typename TI>
std::ostream& fancy_print_array1(std::ostream& os, TI const& ib, TI const& ie, int indent, int tab);

template <typename TI>
std::ostream& fancy_print_array1(std::ostream& os, TI const* d, int num, int indent, int tab);

template <typename V, std::size_t N>
std::ostream& fancy_print(std::ostream& os, const std::array<V, N>& d, int indent, int tab)
{
    os << "[";
    fancy_print_array1(os, &d[0], &d[0] + N, indent + tab, tab);
    os << "]";
    return os;
}

template <typename TI>
std::ostream& fancy_print_key_value(std::ostream& os, TI const& ib, TI const& ie, int indent, int tab);

template <typename TI, typename TFUN>
std::ostream& ContainerOutPut3(std::ostream& os, TI const& ib, TI const& ie, TFUN const& fun, int indent, int tab);

template <typename T>
std::ostream& fancy_print(std::ostream& os, const T& d, int indent, int tab, ENABLE_IF(std::is_arithmetic_v<T>))
{
    os << d;
    return os;
}

template <typename T>
std::ostream& fancy_print(std::ostream& os, T const& d, int indent, int tab, ENABLE_IF((std::rank<T>::value > 0)))
{
    os << "[";
    fancy_print(os, d[0], indent + tab, tab);
    for (size_t i = 1; i < std::extent<T, 0>::value; ++i)
    {
        os << ", ";
        if (std::rank<T>::value > 1)
        {
            os << std::endl
               << std::setw(indent) << " ";
        }
        fancy_print(os, d[i], indent + tab, tab);
    }
    os << std::endl
       << std::setw(indent) << " "
       << "]";

    return os;
}
inline std::ostream& fancy_print(std::ostream& os, std::nullptr_t, int indent, int tab)
{
    os << "<none>";
    return os;
}
inline std::ostream& fancy_print(std::ostream& os, bool const& d, int indent, int tab)
{
    os << std::boolalpha << d;
    return os;
}

inline std::ostream& fancy_print(std::ostream& os, std::string const& s, int indent, int tab)
{
    os << "\"" << s << "\"";
    return os;
}

template <typename T>
std::ostream& fancy_print(std::ostream& os, const std::complex<T>& tv, int indent, int tab)
{
    os << tv.real() << "+" << tv.imag() << "i";
    return (os);
}

template <typename T1, typename T2>
std::ostream& fancy_print(std::ostream& os, std::pair<T1, T2> const& p, int indent, int tab)
{
    fancy_print(os, p.first, indent + tab, tab);
    os << " : ";
    fancy_print(os, p.second, indent + tab, tab);
    return (os);
}

template <typename T1, typename T2>
std::ostream& fancy_print(std::ostream& os, std::map<T1, T2> const& p, int indent, int tab)
{
    os << "{";
    for (auto const& v : p)
    {
        os << std::endl
           << std::setw(indent + 4) << v << ",";
    }
    os << std::endl
       << std::setw(indent) << "}";
    return (os);
}

template <typename U, typename... Others>
std::ostream& fancy_print(std::ostream& os, std::vector<U, Others...> const& d, int indent, int tab)
{
    os << "[ ";
    for (auto const& v : d)
    {
        os << std::endl
           << std::setw(indent + 4) << v << ",";
    }
    os << std::endl
       << std::setw(indent) << " ]";
    return os;
}

template <typename U, typename... Others>
std::ostream& fancy_print(std::ostream& os, std::list<U, Others...> const& d, int indent, int tab)
{
    return fancy_print_array1(os, d.begin(), d.end(), indent);
}

template <typename U, typename... Others>
std::ostream& fancy_print(std::ostream& os, std::set<U, Others...> const& d, int indent, int tab)
{
    return fancy_print_array1(os, d.begin(), d.end(), indent);
}

template <typename U, typename... Others>
std::ostream& fancy_print(std::ostream& os, std::multiset<U, Others...> const& d, int indent, int tab)
{
    return fancy_print_array1(os, d.begin(), d.end(), indent);
}

template <typename TX, typename TY, typename... Others>
std::ostream& fancy_print(std::ostream& os, std::map<TX, TY, Others...> const& d, int indent, int tab)
{
    return fancy_print_key_value(os, d.begin(), d.end(), indent);
}

template <typename TX, typename TY, typename... Others>
std::ostream& fancy_print(std::ostream& os, std::multimap<TX, TY, Others...> const& d, int indent, int tab)
{
    return fancy_print_key_value(os, d.begin(), d.end(), indent);
}
// template <typename T, int... M>
// std::ostream &fancy_print(std::ostream &os, algebra::declare::nTuple_<T, M...> const &v) {
//    return algebra::_detail::printNd_(os, v.m_data_, int_sequence<M...>());
//}

inline std::ostream& fancy_print(std::ostream& os, const std::any& d, int indent, int tab)
{
    os << std::any_cast<std::string>(d);
    return os;
}
template <typename T>
std::ostream& fancy_print(std::ostream& os, const std::shared_ptr<T>& d, int indent, int tab)
{
    os << "[0x" << std::hex << reinterpret_cast<ptrdiff_t>(d.get()) << "]";
    return os;
}

namespace _impl
{
template <typename... Args>
std::ostream& print_helper(std::ostream& os, std::tuple<Args...> const& v, std::integral_constant<int, 0>,
                           int indent = 0, int tab = 1)
{
    return os;
};

template <typename... Args, int N>
std::ostream& print_helper(std::ostream& os, std::tuple<Args...> const& v, std::integral_constant<int, N>,
                           int indent = 0, int tab = 1)
{
    os << ", ";
    fancy_print(os, std::get<sizeof...(Args) - N>(v), indent, tab);
    print_helper(os, v, std::integral_constant<int, N - 1>(), indent, tab);
    return os;
};
} // namespace _impl

template <typename T, typename... Args>
std::ostream& fancy_print(std::ostream& os, std::tuple<T, Args...> const& v, int indent, int tab)
{
    os << "{ ";
    fancy_print(os, std::get<0>(v), indent, tab);
    _impl::print_helper(os, v, std::integral_constant<int, sizeof...(Args)>(), indent + tab, tab);
    os << "}";

    return os;
};

template <typename TI>
std::ostream& fancy_print_array1(std::ostream& os, TI const& ib, TI const& ie, int indent, int tab)
{
    if (ib == ie)
    {
        return os;
    }

    TI it = ib;
    fancy_print(os, *it, indent + tab, tab);
    size_t s = 0;
    ++it;
    for (; it != ie; ++it)
    {
        os << ", ";
        fancy_print(os, *it, indent + tab, tab);
        ++s;
        if (s % 10 == 0)
        {
            os << std::endl;
        }
    }

    return os;
}

// template <typename TI>
// std::ostream &PrintArry1(std::ostream &os, TI const *d, int num, int tab_width) {
//    if (num == 0) { return os; }
//    fancy_print(os, d[0], tab_width + 1);
//    for (int s = 1; s < num; ++s) {
//        os << ", ";
//        fancy_print(os, d[s], tab_width + 1);
//        if (s % 10 == 0) { os << std::endl; }
//    }
//
//    return os;
//}

template <typename TI>
std::ostream& fancy_print_key_value(std::ostream& os, TI const& ib, TI const& ie, int indent, int tab, const std::string& kv_sep = "")
{
    if (ib == ie)
    {
        return os;
    }
    TI it = ib;
    fancy_print(os, it->first, indent);
    os << " " << kv_sep << " ";
    fancy_print(os, it->second, indent + tab, tab);
    ++it;
    for (; it != ie; ++it)
    {
        os << " , " << std::endl
           << std::setw(indent) << " ";
        fancy_print(os, it->first, indent, tab);
        os << " " << kv_sep << " ";
        fancy_print(os, it->second, indent + tab, tab);
    }
    return os;
}

} // namespace sp::utility
namespace std
{
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::complex<T>& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, std::pair<T1, T2> const& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, std::map<T1, T2> const& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
}

template <typename TV, typename... Others>
std::istream& operator>>(std::istream& is, std::vector<TV, Others...>& a)
{
    for (auto it = a.begin(); it != a.end(); ++it)
    {
        is >> *it;
    }
    //    for (auto &v : a) { is >> v; }
    //	std::Duplicate(std::istream_iterator<TV>(is), std::istream_iterator<TV>(),
    // std::back_inserter(a));
    return is;
}

template <typename U, typename... Others>
std::ostream& operator<<(std::ostream& os, std::vector<U, Others...> const& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
}

template <typename U, typename... Others>
std::ostream& operator<<(std::ostream& os, std::list<U, Others...> const& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
}

template <typename U, typename... Others>
std::ostream& operator<<(std::ostream& os, std::set<U, Others...> const& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
}

template <typename U, typename... Others>
std::ostream& operator<<(std::ostream& os, std::multiset<U, Others...> const& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
}

template <typename TX, typename TY, typename... Others>
std::ostream& operator<<(std::ostream& os, std::map<TX, TY, Others...> const& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
}

template <typename TX, typename TY, typename... Others>
std::ostream& operator<<(std::ostream& os, std::multimap<TX, TY, Others...> const& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
}

template <typename T, typename... Args>
std::ostream& operator<<(std::ostream& os, std::tuple<T, Args...> const& v)
{
    return sp::utility::fancy_print(os, v, 0, 4);
};
} // namespace std

#endif /* SP_FANCY_PRINT_H_ */
