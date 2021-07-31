//
// Created by salmon on 18-2-26.
//

#ifndef SIMPLA_SPDMUTILITIES_H
#define SIMPLA_SPDMUTILITIES_H
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "TypeTraits.h"
namespace simpla {
namespace tags {
struct split {
    split(std::size_t _left = 1, std::size_t _right = 1) : my_left(_left), my_right(_right) {}
    ~split() = default;
    virtual std::size_t left() const { return my_left; }
    virtual std::size_t right() const { return my_right; }
    std::size_t my_left = 1, my_right = 1;
};
//! Type enables transmission of splitting proportion from partitioners to range objects
/**
 * In order to make use of such facility Range objects must implement
 * splitting constructor with this type passed and initialize static
 * constant boolean field 'is_splittable_in_proportion' with the value
 * of 'true'
 */
class proportional_split : public split {
   public:
    proportional_split(std::size_t _left = 1, std::size_t _right = 1) {
        split::my_left = _left;
        split::my_right = _right;
    }

    proportional_split(proportional_split const &) = delete;

    ~proportional_split() {}
};
}  // namespace tags

namespace utility {

/**
 *  std::
 */
template <typename T>
void swap(std::complex<T> &lhs, std::complex<T> &rhs) {
    std::swap(lhs, rhs);
}
template <typename T>
void swap(T &lhs, T &rhs, std::enable_if_t<traits::rank<T>::value == 0> *_p = nullptr) {
    std::swap(lhs, rhs);
}
template <typename TL, typename TR>
void swap(TL &&lhs, TR &rhs,
          std::enable_if_t<(traits::is_similar<TL, TR>::value) && (traits::rank<TL>::value > 0)> *_p = nullptr) {
    for (std::size_t i = 0; i < traits::extent<TL>::value; ++i) { swap(lhs[i], rhs[i]); }
}
template <typename TL, typename TR>
void swap(TL &lhs, TR &&rhs,
          std::enable_if_t<(traits::is_similar<TL, TR>::value) && (traits::rank<TL>::value > 0)> *_p = nullptr) {
    for (std::size_t i = 0; i < traits::extent<TL>::value; ++i) { swap(lhs[i], rhs[i]); }
}
template <typename TL, typename TR>
void swap(TL &&lhs, TR &&rhs,
          std::enable_if_t<(traits::is_similar<TL, TR>::value) && (traits::rank<TL>::value > 0)> *_p = nullptr) {
    for (std::size_t i = 0; i < traits::extent<TL>::value; ++i) { swap(lhs[i], rhs[i]); }
}
template <typename TL, typename TR>
void swap(TL &lhs, TR &rhs,
          std::enable_if_t<(traits::is_similar<TL, TR>::value) && (traits::rank<TL>::value > 0)> *_p = nullptr) {
    for (std::size_t i = 0; i < traits::extent<TL>::value; ++i) { swap(lhs[i], rhs[i]); }
}
template <std::size_t I, typename... U>
decltype(auto) get(std::tuple<U...> &v) {
    return std::get<I>(v);
};
template <std::size_t I, typename... U>
decltype(auto) get(std::tuple<U...> const &v) {
    return std::get<I>(v);
};
template <std::size_t I, typename... U>
decltype(auto) get(std::tuple<U...> &&v) {
    return std::get<I>(v);
};

template <std::size_t I, typename U>
decltype(auto) get(U &u, std::enable_if_t<traits::rank<U>::value == 0> *_p = nullptr) {
    return (u);
};
template <std::size_t I, typename U>
decltype(auto) get(U &&u, std::enable_if_t<traits::rank<U>::value == 0> *_p = nullptr) {
    return std::move(u);
};
template <std::size_t I, typename U>
decltype(auto) get(U &u, std::enable_if_t<(traits::rank<U>::value > 0)> *_p = nullptr) {
    static_assert(I < traits::extent<U>::value, "out of range");
    return u[I];
};
template <std::size_t I, typename U>
decltype(auto) get(U &&u, std::enable_if_t<(traits::rank<U>::value > 0)> *_p = nullptr) {
    static_assert(I < traits::extent<U>::value, "out of range");
    return std::move(u[I]);
};

template <typename V>
auto sum_n(V const &v) {
    return v;
}
template <typename V, typename... Others>
auto sum_n(V const &v, Others &&... others) {
    return v + sum_n(std::forward<Others>(others)...);
}
template <typename V>
auto product_n(V const &v) {
    return v;
}
template <typename V, typename... Others>
auto product_n(V const &v, Others &&... others) {
    return v * product_n(std::forward<Others>(others)...);
}
/**
 *  not std::
 */
template <typename T, typename V>
V Fill(T &v, V start, V inc = 0,
       std::enable_if_t<traits::rank<T>::value == 0 && !traits::is_range<T>::value> *_p = nullptr) {
    v = start;
    return start + inc;
}

template <typename T, typename V>
V Fill(T &v, V start, V inc = 0, std::enable_if_t<(traits::rank<T>::value > 0)> *_p = nullptr) {
    for (decltype(auto) item : v) { start = Fill(item, start, inc); }
    return start;
}

template <typename TL, typename TR>
bool IsEqual(TL const &lhs, TR const &rhs,
             std::enable_if_t<!traits::is_similar<TL, TR>::value && traits::rank<TR>::value != 0> *_p = nullptr) {
    return false;
}
template <typename TL, typename TR>
bool IsEqual(TL const &lhs, TR const &rhs,
             std::enable_if_t<!traits::is_similar<TL, TR>::value && traits::rank<TR>::value == 0> *_p = nullptr) {
    bool res = true;
    for (auto const &item : lhs) {
        res = res && IsEqual(item, rhs);
        if (!res) { break; }
    }
    return res;
}
template <typename TL, typename TR>
bool IsEqual(TL const &lhs, TR const &rhs,
             std::enable_if_t<traits::is_similar<TL, TR>::value && traits::rank<TR>::value == 0> *_p = nullptr) {
    return lhs == rhs;
}
template <typename TL, typename TR>
bool IsEqual(TL const &lhs, TR const &rhs,
             std::enable_if_t<traits::is_similar<TL, TR>::value && traits::rank<TR>::value != 0> *_p = nullptr) {
    int n = 0;
    bool res = true;
    for (auto const &item : lhs) {
        res = res && IsEqual(item, rhs[n]);
        ++n;
        if (!res) { break; }
    }
    return res;
}
template <typename TL, typename TR>
bool IsEqualP(TL const &lhs, TR const *rhs, std::enable_if_t<traits::rank<TL>::value == 0> *_p = nullptr) {
    return lhs == *rhs;
}
template <typename TL, typename TR>
bool IsEqualP(TL const &lhs, TR const *rhs, std::enable_if_t<traits::rank<TL>::value != 0> *_p = nullptr) {
    bool res = true;

    for (auto const &item : lhs) {
        res = res && IsEqualP(item, rhs);
        rhs += traits::number_of_elements<traits::remove_extent_t<TL>>::value;
        if (!res) { break; }
    }
    return res;
}
template <typename OS, typename TV>
OS &FancyPrint(OS &os, TV const &t, int indent = 0, unsigned tab_width = 1, char left = '{', char split = ',',
               char right = '}', std::enable_if_t<(traits::rank<TV>::value == 0)> *_p = nullptr) {
    os << t;
    return os;
};
template <typename OS, typename TV>
OS &FancyPrint(OS &os, TV const &t, int indent = 0, unsigned tab_width = 1, char left = '{', char split = ',',
               char right = '}', std::enable_if_t<(traits::rank<TV>::value != 0)> *_p = nullptr) {
    os << std::setw(indent + tab_width) << left;
    if (traits::rank<TV>::value > 1) { os << std::endl; }
    bool is_first = true;
    for (decltype(auto) item : t) {
        if (is_first) {
            is_first = false;
        } else {
            os << split;
            if (traits::rank<TV>::value > 1) { os << std::endl; }
        }
        FancyPrint(os, item, indent + tab_width, tab_width, left, split, right);
    }
    if (traits::rank<TV>::value > 1) { os << std::endl << std::setw(indent + 1); }
    os << right;
    return os;
};

template <typename OS, typename TV, typename TI>
TV FancyPrintP(OS &os, TV v, unsigned int ndim, TI const *dims, int indent = 0, unsigned tab_width = 1, char left = '{',
               char split = ',', char right = '}') {
    os << left;
    if (ndim > 1) { os << std::endl; }
    for (std::size_t s = 0; s < dims[0]; ++s) {
        if (s > 0) {
            if (ndim > 1) {
                os << split << std::endl << std::setw(indent + tab_width);
            } else {
                os << split << std::setw(tab_width);
            }
        } else if (ndim > 1) {
            os << std::setw(indent + tab_width);
        }

        if (ndim == 1) {
            os << *v;
            ++v;
        } else {
            v = FancyPrintP(os, v, ndim - 1, dims + 1, indent + tab_width, tab_width, left, split, right);
        }
    }
    if (ndim > 1) { os << std::endl << std::setw(indent); }
    os << right;
    return v;
};

/** @ingroup design_pattern
 *
 * @addtogroup  singleton Singleton
 * @{
 *
 * @brief singleton
 *
 * @note  Meyers Singleton，
 * Ref:Andrei Alexandrescu Chap 6.4
 * Modern C++ Design Generic Programming and Design Patterns Applied 2001 Addison Wesley ,
 */
template <class T>
class SingletonHolder {
   public:
    static T &instance() {
        if (!pInstance_) {
            //#pragma omp critical
            // TOD add some for mt critical
            if (!pInstance_) {
                static T tmp;
                pInstance_ = &tmp;
            }
        }
        return *pInstance_;
    }

   protected:
    SingletonHolder() {}
    ~SingletonHolder() {}
    static T *volatile pInstance_;
};

template <class T>
T *volatile SingletonHolder<T>::pInstance_ = 0;

}  //    namespace utility {
}  // namespace simpla {
#define SP_FILE_LINE_STAMP \
    (std::string("[ ") + (__FILE__) + ":" + std::to_string(__LINE__) + ":0: " + (__PRETTY_FUNCTION__) + " ] :")

#define ERR_UNIMPLEMENTED throw(std::runtime_error("\nERROR:" + SP_FILE_LINE_STAMP + "UNIMPLEMENTED!"))
#define ERR_OUT_OF_RANGE(_MSG_) throw(std::out_of_range("\nERROR:" + SP_FILE_LINE_STAMP + "OUT_OF_RANGE!" + _MSG_))

namespace simpla {

/**
 * @ingroup toolbox
 * @addtogroup fancy_print Fancy print
 * @{
 */

template <typename TV, typename TI>
inline TV const *printNdArray(std::ostream &os, TV const *v, int rank, TI const *d, bool is_first = true,
                              bool is_last = true, std::string const &left_brace = "{", std::string const &sep = ",",
                              std::string const &right_brace = "}", bool is_slow_first = true, int tab_width = 0,
                              int indent = 0) {
    constexpr int ELE_NUM_PER_LINE = 10;
    if (rank == 1) {
        os << left_brace;
        for (int s = 0; s < d[0]; ++s) {
            if (s > 0) { os << sep; }
            if (s % ELE_NUM_PER_LINE == 0 && s != 0) { os << std::endl; }
            if (tab_width > 0) { os << std::setw(10) << " "; }
            os << (*v);
            ++v;
        }
        os << right_brace;

    } else {
        os << left_brace;

        for (int s = 0; s < d[0]; ++s) {
            if (s > 0) {
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

namespace detail {
template <std::size_t... IDX, typename V, typename I>
auto GetValue(std::integer_sequence<std::size_t, IDX...>, V const &v, I const *idx) {
    return v(idx[IDX]...);
};
}

template <int NDIMS, typename TV, typename TI>
std::ostream &FancyPrintNdSlowFirst(std::ostream &os, TV const &v, int depth, int *idx, TI const *lo, TI const *hi,
                                    int indent = 0, int tab_width = 0, std::string const &left_brace = "{",
                                    std::string const &sep = ",", std::string const &right_brace = "}") {
    if (depth >= NDIMS) { return os; }

    if (depth == NDIMS - 1) {
        os << std::setw(indent) << left_brace;
        idx[NDIMS - 1] = lo[NDIMS - 1];
        os << std::setw(tab_width) << detail::GetValue(std::make_index_sequence<NDIMS>(), v, idx);
        for (idx[NDIMS - 1] = lo[NDIMS - 1] + 1; idx[NDIMS - 1] < hi[NDIMS - 1]; ++idx[NDIMS - 1]) {
            os << sep << std::setw(tab_width) << detail::GetValue(std::make_index_sequence<NDIMS>(), v, idx);
        }
        os << right_brace;
    } else {
        os << std::setw(indent) << left_brace << std::endl;
        idx[depth] = lo[depth];
        FancyPrintNdSlowFirst<NDIMS>(os, v, depth + 1, idx, lo, hi, indent + 1, tab_width, left_brace, sep,
                                     right_brace);

        for (idx[depth] = lo[depth] + 1; idx[depth] < hi[depth]; ++idx[depth]) {
            os << sep << std::endl;
            FancyPrintNdSlowFirst<NDIMS>(os, v, depth + 1, idx, lo, hi, indent + 1, tab_width, left_brace, sep,
                                         right_brace);
        }
        os << std::endl << std::setw(indent) << right_brace;
    }

    return os;
}

template <int NDIMS, typename TV, typename TI>
std::ostream &FancyPrintNdFastFirst(std::ostream &os, TV const &v, int depth, int *idx, TI const *lo, TI const *hi,
                                    int indent = 0, int tab_width = 0, std::string const &left_brace = "{",
                                    std::string const &sep = ",", std::string const &right_brace = "}") {
    if (depth < 0) { return os; }

    if (depth == 0) {
        os << std::setw(indent + depth) << left_brace;
        idx[0] = lo[0];
        os << std::setw(tab_width) << detail::GetValue(std::make_index_sequence<NDIMS>(), v, idx);
        for (idx[0] = lo[0] + 1; idx[0] < hi[0]; ++idx[0]) {
            os << "," << std::setw(tab_width) << detail::GetValue(std::make_index_sequence<NDIMS>(), v, idx);
        }
        os << right_brace;
    } else {
        os << std::setw(indent) << left_brace << std::endl;
        idx[depth] = lo[depth];
        FancyPrintNdFastFirst<NDIMS>(os, v, depth - 1, idx, lo, hi, indent + 1, tab_width, left_brace, sep,
                                     right_brace);

        for (idx[depth] = lo[depth] + 1; idx[depth] < hi[depth]; ++idx[depth]) {
            os << "," << std::endl;
            FancyPrintNdFastFirst<NDIMS>(os, v, depth - 1, idx, lo, hi, indent + 1, tab_width, left_brace, sep,
                                         right_brace);
        }
        os << std::endl << std::setw(indent) << right_brace;
    }

    return os;
}

template <int NDIMS, typename TV, typename TI>
std::ostream &FancyPrintNd(std::ostream &os, TV const &v, TI const *lo, TI const *hi, bool is_slow_first = true,
                           int indent = 0, int tab_width = 8, std::string const &left_brace = "{",
                           std::string const &sep = ",", std::string const &right_brace = "}") {
    constexpr int ELE_NUM_PER_LINE = 10;

    int idx[NDIMS];
    if (is_slow_first) {
        FancyPrintNdSlowFirst<NDIMS>(os, v, 0, idx, lo, hi, indent, tab_width, left_brace, sep, right_brace);
    } else {
        FancyPrintNdFastFirst<NDIMS>(os, v, NDIMS - 1, idx, lo, hi, indent, tab_width, left_brace, sep, right_brace);
    }

    return os;
}
template <typename TX, typename TY, typename... Others>
std::istream &get_(std::istream &is, size_t num, std::map<TX, TY, Others...> &a) {
    for (size_t s = 0; s < num; ++s) {
        TX x;
        TY y;
        is >> x >> y;
        a.emplace(x, y);
    }
    return is;
}
template <typename V>
std::ostream &FancyPrint(std::ostream &os, V const *d, std::size_t ndims, std::size_t const *extents, int indent) {
    printNdArray(os, d, ndims, extents, true, false, "[", ",", "]", true, 0, indent);
    return os;
}

template <typename TI>
std::ostream &PrintArray1(std::ostream &os, TI const &ib, TI const &ie, int indent = 0);

template <typename TI>
std::ostream &PrintArray1(std::ostream &os, TI const *d, int num, int indent = 0);

template <typename TI>
std::ostream &PrintKeyValue(std::ostream &os, TI const &ib, TI const &ie, int indent = 0);

template <typename TI, typename TFUN>
std::ostream &ContainerOutPut3(std::ostream &os, TI const &ib, TI const &ie, TFUN const &fun, int indent = 0);

template <typename T>
std::ostream &FancyPrint(std::ostream &os, T const &d, int indent,
                         std::enable_if_t<(traits::rank<T>::value == 0)> *_p = nullptr) {
    os << d;
    return os;
}

template <typename T>
std::ostream &FancyPrint(std::ostream &os, T const &d, int indent,
                         std::enable_if_t<(traits::rank<T>::value > 0)> *_p = nullptr) {
    os << "[";
    FancyPrint(os, d[0], indent + 1);
    for (size_t i = 1; i < std::extent<T, 0>::value; ++i) {
        os << ", ";
        if (std::rank<T>::value > 1) { os << std::endl << std::setw(indent) << " "; }
        FancyPrint(os, d[i], indent + 1);
    }
    os << "]";

    return os;
}
inline std::ostream &FancyPrint(std::ostream &os, bool const &d, int indent) {
    os << std::boolalpha << d;
    return os;
}
inline std::ostream &FancyPrint(std::ostream &os, std::string const &s, int indent) {
    os << "\"" << s << "\"";
    return os;
}

template <typename T>
std::ostream &FancyPrint(std::ostream &os, const std::complex<T> &tv, int indent = 0) {
    os << "{" << tv.real() << "," << tv.imag() << "}";
    return (os);
}

template <typename T1, typename T2>
std::ostream &FancyPrint(std::ostream &os, std::pair<T1, T2> const &p, int indent = 0) {
    os << p.first << " = { " << p.second << " }";
    return (os);
}

template <typename T1, typename T2>
std::ostream &FancyPrint(std::ostream &os, std::map<T1, T2> const &p, int indent = 0) {
    for (auto const &v : p) os << v << "," << std::endl;
    return (os);
}

template <typename U, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::vector<U, Others...> const &d, int indent = 0) {
    os << "[ ";
    PrintArray1(os, d.begin(), d.end(), indent);
    os << " ]";
    return os;
}

template <typename U, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::list<U, Others...> const &d, int indent = 0) {
    return PrintArray1(os, d.begin(), d.end(), indent);
}

template <typename U, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::set<U, Others...> const &d, int indent = 0) {
    return PrintArray1(os, d.begin(), d.end(), indent);
}

template <typename U, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::multiset<U, Others...> const &d, int indent = 0) {
    return PrintArray1(os, d.begin(), d.end(), indent);
}

template <typename TX, typename TY, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::map<TX, TY, Others...> const &d, int indent = 0) {
    return PrintKeyValue(os, d.begin(), d.end(), indent);
}

template <typename TX, typename TY, typename... Others>
std::ostream &FancyPrint(std::ostream &os, std::multimap<TX, TY, Others...> const &d, int indent = 0) {
    return PrintKeyValue(os, d.begin(), d.end(), indent);
}
// template <typename T, int... M>
// std::ostream &FancyPrint(std::ostream &os, algebra::declare::nTuple_<T, M...> const &v) {
//    return algebra::_detail::printNd_(os, v.m_data_, int_sequence<M...>());
//}
namespace _impl {
template <typename... Args>
std::ostream &print_helper(std::ostream &os, std::tuple<Args...> const &v, std::integral_constant<int, 0>,
                           int indent = 0) {
    return os;
};

template <typename... Args, int N>
std::ostream &print_helper(std::ostream &os, std::tuple<Args...> const &v, std::integral_constant<int, N>,
                           int indent = 0) {
    os << ", ";
    FancyPrint(os, std::get<sizeof...(Args) - N>(v), indent);
    print_helper(os, v, std::integral_constant<int, N - 1>(), indent);
    return os;
};
}

template <typename T, typename... Args>
std::ostream &FancyPrint(std::ostream &os, std::tuple<T, Args...> const &v, int indent = 0) {
    os << "{ ";
    FancyPrint(os, std::get<0>(v), indent);
    _impl::print_helper(os, v, std::integral_constant<int, sizeof...(Args)>(), indent + 1);
    os << "}";

    return os;
};

template <typename TI>
std::ostream &PrintArray1(std::ostream &os, TI const &ib, TI const &ie, int indent) {
    if (ib == ie) { return os; }

    TI it = ib;
    FancyPrint(os, *it, indent + 1);
    size_t s = 0;
    ++it;
    for (; it != ie; ++it) {
        os << ", ";
        FancyPrint(os, *it, indent + 1);
        ++s;
        if (s % 10 == 0) { os << std::endl; }
    }

    return os;
}

// template <typename TI>
// std::ostream &PrintArry1(std::ostream &os, TI const *d, int num, int tab_width) {
//    if (num == 0) { return os; }
//    FancyPrint(os, d[0], tab_width + 1);
//    for (int s = 1; s < num; ++s) {
//        os << ", ";
//        FancyPrint(os, d[s], tab_width + 1);
//        if (s % 10 == 0) { os << std::endl; }
//    }
//
//    return os;
//}

template <typename TI>
std::ostream &PrintKeyValue(std::ostream &os, TI const &ib, TI const &ie, int indent) {
    if (ib == ie) { return os; }
    TI it = ib;
    FancyPrint(os, it->first, indent);
    os << "=";
    FancyPrint(os, it->second, indent + 1);
    ++it;
    for (; it != ie; ++it) {
        os << " , " << std::endl << std::setw(indent) << " ";
        FancyPrint(os, it->first, indent);
        os << " = ";
        FancyPrint(os, it->second, indent + 1);
    }
    return os;
}

std::size_t MakeUUID();

}  // namespace simpla
namespace std {
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::complex<T> &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, std::pair<T1, T2> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, std::map<T1, T2> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename TV, typename... Others>
std::istream &operator>>(std::istream &is, std::vector<TV, Others...> &a) {
    for (auto it = a.begin(); it != a.end(); ++it) { is >> *it; }
    //    for (auto &v : a) { is >> v; }
    //	std::Duplicate(std::istream_iterator<TV>(is), std::istream_iterator<TV>(),
    // std::back_inserter(a));
    return is;
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::vector<U, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::list<U, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::set<U, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename U, typename... Others>
std::ostream &operator<<(std::ostream &os, std::multiset<U, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename TX, typename TY, typename... Others>
std::ostream &operator<<(std::ostream &os, std::map<TX, TY, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename TX, typename TY, typename... Others>
std::ostream &operator<<(std::ostream &os, std::multimap<TX, TY, Others...> const &v) {
    return simpla::FancyPrint(os, v, 0);
}

template <typename T, typename... Args>
std::ostream &operator<<(std::ostream &os, std::tuple<T, Args...> const &v) {
    return simpla::FancyPrint(os, v, 0);
};
}  // namespace std{

#endif  // SIMPLA_SPDMUTILITIES_H
