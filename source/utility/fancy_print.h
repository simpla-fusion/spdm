//
// Created by salmon on 18-2-26.
//

#ifndef SP_FANCY_PRINT_H_
#define SP_FANCY_PRINT_H_
#include <complex>
#include <cstddef>
#include <iomanip>
#include <list>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
namespace sp
{

    template <typename OS, typename TV>
    OS &fancyPrint(OS &os, TV const &t, int indent = 0, unsigned tab_width = 1, char left = '{', char split = ',',
                   char right = '}', std::enable_if_t<(traits::rank<TV>::value == 0)> *_p = nullptr)
    {
        os << t;
        return os;
    };
    template <typename OS, typename TV>
    OS &fancyPrint(OS &os, TV const &t, int indent = 0, unsigned tab_width = 1, char left = '{', char split = ',',
                   char right = '}', std::enable_if_t<(traits::rank<TV>::value != 0)> *_p = nullptr)
    {
        os << std::setw(indent + tab_width) << left;
        if (traits::rank<TV>::value > 1)
        {
            os << std::endl;
        }
        bool is_first = true;
        for (decltype(auto) item : t)
        {
            if (is_first)
            {
                is_first = false;
            }
            else
            {
                os << split;
                if (traits::rank<TV>::value > 1)
                {
                    os << std::endl;
                }
            }
            fancyPrint(os, item, indent + tab_width, tab_width, left, split, right);
        }
        if (traits::rank<TV>::value > 1)
        {
            os << std::endl
               << std::setw(indent + 1);
        }
        os << right;
        return os;
    };

    template <typename OS, typename TV, typename TI>
    TV fancyPrintP(OS &os, TV v, unsigned int ndim, TI const *dims, int indent = 0, unsigned tab_width = 1, char left = '{',
                   char split = ',', char right = '}')
    {
        os << left;
        if (ndim > 1)
        {
            os << std::endl;
        }
        for (std::size_t s = 0; s < dims[0]; ++s)
        {
            if (s > 0)
            {
                if (ndim > 1)
                {
                    os << split << std::endl
                       << std::setw(indent + tab_width);
                }
                else
                {
                    os << split << std::setw(tab_width);
                }
            }
            else if (ndim > 1)
            {
                os << std::setw(indent + tab_width);
            }

            if (ndim == 1)
            {
                os << *v;
                ++v;
            }
            else
            {
                v = fancyPrintP(os, v, ndim - 1, dims + 1, indent + tab_width, tab_width, left, split, right);
            }
        }
        if (ndim > 1)
        {
            os << std::endl
               << std::setw(indent);
        }
        os << right;
        return v;
    };
} // namespace sp
#define SP_FILE_LINE_STAMP \
    (std::string("[ ") + (__FILE__) + ":" + std::to_string(__LINE__) + ":0: " + (__PRETTY_FUNCTION__) + " ] :")

#define ERR_UNIMPLEMENTED throw(std::runtime_error("\nERROR:" + SP_FILE_LINE_STAMP + "UNIMPLEMENTED!"))
#define ERR_OUT_OF_RANGE(_MSG_) throw(std::out_of_range("\nERROR:" + SP_FILE_LINE_STAMP + "OUT_OF_RANGE!" + _MSG_))

namespace sp
{

    /**
     * @ingroup toolbox
     * @addtogroup fancy_print Fancy print
     * @{
     */

    template <typename TV, typename TI>
    inline TV const *printNdArray(std::ostream &os, TV const *v, int rank, TI const *d, bool is_first = true,
                                  bool is_last = true, std::string const &left_brace = "{", std::string const &sep = ",",
                                  std::string const &right_brace = "}", bool is_slow_first = true, int tab_width = 0,
                                  int indent = 0)
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
                v = printNdArray(os, v, rank - 1, d + 1, s == 0, s == d[0] - 1, left_brace, sep, right_brace, is_slow_first,
                                 tab_width, indent + 1);
            }
            os << right_brace;
            return (v);
        }
        //    if (is_last) { os << std::endl; }
        return v;
    }

    namespace detail
    {
        template <std::size_t... IDX, typename V, typename I>
        auto GetValue(std::integer_sequence<std::size_t, IDX...>, V const &v, I const *idx)
        {
            return v(idx[IDX]...);
        };
    } // namespace detail

    template <int NDIMS, typename TV, typename TI>
    std::ostream &fancyPrintNdSlowFirst(std::ostream &os, TV const &v, int depth, int *idx, TI const *lo, TI const *hi,
                                        int indent = 0, int tab_width = 0, std::string const &left_brace = "{",
                                        std::string const &sep = ",", std::string const &right_brace = "}")
    {
        if (depth >= NDIMS)
        {
            return os;
        }

        if (depth == NDIMS - 1)
        {
            os << std::setw(indent) << left_brace;
            idx[NDIMS - 1] = lo[NDIMS - 1];
            os << std::setw(tab_width) << detail::GetValue(std::make_index_sequence<NDIMS>(), v, idx);
            for (idx[NDIMS - 1] = lo[NDIMS - 1] + 1; idx[NDIMS - 1] < hi[NDIMS - 1]; ++idx[NDIMS - 1])
            {
                os << sep << std::setw(tab_width) << detail::GetValue(std::make_index_sequence<NDIMS>(), v, idx);
            }
            os << right_brace;
        }
        else
        {
            os << std::setw(indent) << left_brace << std::endl;
            idx[depth] = lo[depth];
            fancyPrintNdSlowFirst<NDIMS>(os, v, depth + 1, idx, lo, hi, indent + 1, tab_width, left_brace, sep,
                                         right_brace);

            for (idx[depth] = lo[depth] + 1; idx[depth] < hi[depth]; ++idx[depth])
            {
                os << sep << std::endl;
                fancyPrintNdSlowFirst<NDIMS>(os, v, depth + 1, idx, lo, hi, indent + 1, tab_width, left_brace, sep,
                                             right_brace);
            }
            os << std::endl
               << std::setw(indent) << right_brace;
        }

        return os;
    }

    template <int NDIMS, typename TV, typename TI>
    std::ostream &fancyPrintNdFastFirst(std::ostream &os, TV const &v, int depth, int *idx, TI const *lo, TI const *hi,
                                        int indent = 0, int tab_width = 0, std::string const &left_brace = "{",
                                        std::string const &sep = ",", std::string const &right_brace = "}")
    {
        if (depth < 0)
        {
            return os;
        }

        if (depth == 0)
        {
            os << std::setw(indent + depth) << left_brace;
            idx[0] = lo[0];
            os << std::setw(tab_width) << detail::GetValue(std::make_index_sequence<NDIMS>(), v, idx);
            for (idx[0] = lo[0] + 1; idx[0] < hi[0]; ++idx[0])
            {
                os << "," << std::setw(tab_width) << detail::GetValue(std::make_index_sequence<NDIMS>(), v, idx);
            }
            os << right_brace;
        }
        else
        {
            os << std::setw(indent) << left_brace << std::endl;
            idx[depth] = lo[depth];
            fancyPrintNdFastFirst<NDIMS>(os, v, depth - 1, idx, lo, hi, indent + 1, tab_width, left_brace, sep,
                                         right_brace);

            for (idx[depth] = lo[depth] + 1; idx[depth] < hi[depth]; ++idx[depth])
            {
                os << "," << std::endl;
                fancyPrintNdFastFirst<NDIMS>(os, v, depth - 1, idx, lo, hi, indent + 1, tab_width, left_brace, sep,
                                             right_brace);
            }
            os << std::endl
               << std::setw(indent) << right_brace;
        }

        return os;
    }

    template <int NDIMS, typename TV, typename TI>
    std::ostream &fancyPrintNd(std::ostream &os, TV const &v, TI const *lo, TI const *hi, bool is_slow_first = true,
                               int indent = 0, int tab_width = 8, std::string const &left_brace = "{",
                               std::string const &sep = ",", std::string const &right_brace = "}")
    {
        constexpr int ELE_NUM_PER_LINE = 10;

        int idx[NDIMS];
        if (is_slow_first)
        {
            fancyPrintNdSlowFirst<NDIMS>(os, v, 0, idx, lo, hi, indent, tab_width, left_brace, sep, right_brace);
        }
        else
        {
            fancyPrintNdFastFirst<NDIMS>(os, v, NDIMS - 1, idx, lo, hi, indent, tab_width, left_brace, sep, right_brace);
        }

        return os;
    }
    template <typename TX, typename TY, typename... Others>
    std::istream &get_(std::istream &is, size_t num, std::map<TX, TY, Others...> &a)
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
    std::ostream &fancyPrint(std::ostream &os, V const *d, std::size_t ndims, std::size_t const *extents, int indent)
    {
        printNdArray(os, d, ndims, extents, true, false, "[", ",", "]", true, 0, indent);
        return os;
    }

    template <typename TI>
    std::ostream &printArray1(std::ostream &os, TI const &ib, TI const &ie, int indent = 0);

    template <typename TI>
    std::ostream &printArray1(std::ostream &os, TI const *d, int num, int indent = 0);

    template <typename TI>
    std::ostream &PrintKeyValue(std::ostream &os, TI const &ib, TI const &ie, int indent = 0);

    template <typename TI, typename TFUN>
    std::ostream &ContainerOutPut3(std::ostream &os, TI const &ib, TI const &ie, TFUN const &fun, int indent = 0);

    template <typename T>
    std::ostream &fancyPrint(std::ostream &os, T const &d, int indent,
                             std::enable_if_t<(traits::rank<T>::value == 0)> *_p = nullptr)
    {
        os << d;
        return os;
    }

    template <typename T>
    std::ostream &fancyPrint(std::ostream &os, T const &d, int indent,
                             std::enable_if_t<(traits::rank<T>::value > 0)> *_p = nullptr)
    {
        os << "[";
        fancyPrint(os, d[0], indent + 1);
        for (size_t i = 1; i < std::extent<T, 0>::value; ++i)
        {
            os << ", ";
            if (std::rank<T>::value > 1)
            {
                os << std::endl
                   << std::setw(indent) << " ";
            }
            fancyPrint(os, d[i], indent + 1);
        }
        os << "]";

        return os;
    }
   
    inline std::ostream &fancyPrint(std::ostream &os, bool const &d, int indent)
    {
        os << std::boolalpha << d;
        return os;
    }
   
    inline std::ostream &fancyPrint(std::ostream &os, std::string const &s, int indent)
    {
        os << "\"" << s << "\"";
        return os;
    }

    template <typename T>
    std::ostream &fancyPrint(std::ostream &os, const std::complex<T> &tv, int indent = 0)
    {
        os << "{" << tv.real() << "," << tv.imag() << "}";
        return (os);
    }

    template <typename T1, typename T2>
    std::ostream &fancyPrint(std::ostream &os, std::pair<T1, T2> const &p, int indent = 0)
    {
        os << p.first << " = { " << p.second << " }";
        return (os);
    }

    template <typename T1, typename T2>
    std::ostream &fancyPrint(std::ostream &os, std::map<T1, T2> const &p, int indent = 0)
    {
        for (auto const &v : p)
            os << v << "," << std::endl;
        return (os);
    }

    template <typename U, typename... Others>
    std::ostream &fancyPrint(std::ostream &os, std::vector<U, Others...> const &d, int indent = 0)
    {
        os << "[ ";
        printArray1(os, d.begin(), d.end(), indent);
        os << " ]";
        return os;
    }

    template <typename U, typename... Others>
    std::ostream &fancyPrint(std::ostream &os, std::list<U, Others...> const &d, int indent = 0)
    {
        return printArray1(os, d.begin(), d.end(), indent);
    }

    template <typename U, typename... Others>
    std::ostream &fancyPrint(std::ostream &os, std::set<U, Others...> const &d, int indent = 0)
    {
        return printArray1(os, d.begin(), d.end(), indent);
    }

    template <typename U, typename... Others>
    std::ostream &fancyPrint(std::ostream &os, std::multiset<U, Others...> const &d, int indent = 0)
    {
        return printArray1(os, d.begin(), d.end(), indent);
    }

    template <typename TX, typename TY, typename... Others>
    std::ostream &fancyPrint(std::ostream &os, std::map<TX, TY, Others...> const &d, int indent = 0)
    {
        return PrintKeyValue(os, d.begin(), d.end(), indent);
    }

    template <typename TX, typename TY, typename... Others>
    std::ostream &fancyPrint(std::ostream &os, std::multimap<TX, TY, Others...> const &d, int indent = 0)
    {
        return PrintKeyValue(os, d.begin(), d.end(), indent);
    }
    // template <typename T, int... M>
    // std::ostream &fancyPrint(std::ostream &os, algebra::declare::nTuple_<T, M...> const &v) {
    //    return algebra::_detail::printNd_(os, v.m_data_, int_sequence<M...>());
    //}
    namespace _impl
    {
        template <typename... Args>
        std::ostream &print_helper(std::ostream &os, std::tuple<Args...> const &v, std::integral_constant<int, 0>,
                                   int indent = 0)
        {
            return os;
        };

        template <typename... Args, int N>
        std::ostream &print_helper(std::ostream &os, std::tuple<Args...> const &v, std::integral_constant<int, N>,
                                   int indent = 0)
        {
            os << ", ";
            fancyPrint(os, std::get<sizeof...(Args) - N>(v), indent);
            print_helper(os, v, std::integral_constant<int, N - 1>(), indent);
            return os;
        };
    } // namespace _impl

    template <typename T, typename... Args>
    std::ostream &fancyPrint(std::ostream &os, std::tuple<T, Args...> const &v, int indent = 0)
    {
        os << "{ ";
        fancyPrint(os, std::get<0>(v), indent);
        _impl::print_helper(os, v, std::integral_constant<int, sizeof...(Args)>(), indent + 1);
        os << "}";

        return os;
    };

    template <typename TI>
    std::ostream &printArray1(std::ostream &os, TI const &ib, TI const &ie, int indent)
    {
        if (ib == ie)
        {
            return os;
        }

        TI it = ib;
        fancyPrint(os, *it, indent + 1);
        size_t s = 0;
        ++it;
        for (; it != ie; ++it)
        {
            os << ", ";
            fancyPrint(os, *it, indent + 1);
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
    //    fancyPrint(os, d[0], tab_width + 1);
    //    for (int s = 1; s < num; ++s) {
    //        os << ", ";
    //        fancyPrint(os, d[s], tab_width + 1);
    //        if (s % 10 == 0) { os << std::endl; }
    //    }
    //
    //    return os;
    //}

    template <typename TI>
    std::ostream &printKeyValue(std::ostream &os, TI const &ib, TI const &ie, int indent)
    {
        if (ib == ie)
        {
            return os;
        }
        TI it = ib;
        fancyPrint(os, it->first, indent);
        os << "=";
        fancyPrint(os, it->second, indent + 1);
        ++it;
        for (; it != ie; ++it)
        {
            os << " , " << std::endl
               << std::setw(indent) << " ";
            fancyPrint(os, it->first, indent);
            os << " = ";
            fancyPrint(os, it->second, indent + 1);
        }
        return os;
    }


} // namespace sp
namespace std
{
    template <typename T>
    std::ostream &operator<<(std::ostream &os, const std::complex<T> &v)
    {
        return sp::fancyPrint(os, v, 0);
    }

    template <typename T1, typename T2>
    std::ostream &operator<<(std::ostream &os, std::pair<T1, T2> const &v)
    {
        return sp::fancyPrint(os, v, 0);
    }

    template <typename T1, typename T2>
    std::ostream &operator<<(std::ostream &os, std::map<T1, T2> const &v)
    {
        return sp::fancyPrint(os, v, 0);
    }

    template <typename TV, typename... Others>
    std::istream &operator>>(std::istream &is, std::vector<TV, Others...> &a)
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
    std::ostream &operator<<(std::ostream &os, std::vector<U, Others...> const &v)
    {
        return sp::fancyPrint(os, v, 0);
    }

    template <typename U, typename... Others>
    std::ostream &operator<<(std::ostream &os, std::list<U, Others...> const &v)
    {
        return sp::fancyPrint(os, v, 0);
    }

    template <typename U, typename... Others>
    std::ostream &operator<<(std::ostream &os, std::set<U, Others...> const &v)
    {
        return sp::fancyPrint(os, v, 0);
    }

    template <typename U, typename... Others>
    std::ostream &operator<<(std::ostream &os, std::multiset<U, Others...> const &v)
    {
        return sp::fancyPrint(os, v, 0);
    }

    template <typename TX, typename TY, typename... Others>
    std::ostream &operator<<(std::ostream &os, std::map<TX, TY, Others...> const &v)
    {
        return sp::fancyPrint(os, v, 0);
    }

    template <typename TX, typename TY, typename... Others>
    std::ostream &operator<<(std::ostream &os, std::multimap<TX, TY, Others...> const &v)
    {
        return sp::fancyPrint(os, v, 0);
    }

    template <typename T, typename... Args>
    std::ostream &operator<<(std::ostream &os, std::tuple<T, Args...> const &v)
    {
        return sp::fancyPrint(os, v, 0);
    };
} // namespace std

#endif // SP_FANCY_PRINT_H_
