//
// Created by salmon on 18-2-27.
//

#ifndef SP_TYPETRAITS_H_
#define SP_TYPETRAITS_H_

#include <any>
#include <array>
#include <complex>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
namespace sp
{
namespace traits
{

template <typename V>
V const& as_const(V& v)
{
    return v;
}
template <typename V>
V const& as_const(V const& v)
{
    return v;
}
template <typename V>
V const* as_const(V* v)
{
    return v;
}
template <typename V>
V const* as_const(V const* v)
{
    return v;
}

template <typename U>
U* remove_const(U* d)
{
    return d;
}
template <typename U>
U* remove_const(U const* d)
{
    return const_cast<U*>(d);
}
template <typename U>
U& remove_const(U& d)
{
    return d;
}
template <typename U>
U& remove_const(U const& d)
{
    return const_cast<U&>(d);
}

template <typename V, typename SFINAE = void>
struct type_name
{
    static std::string str() { return typeid(V).name(); }
};
template <>
struct type_name<double>
{
    static std::string str() { return "double"; }
};
template <>
struct type_name<int>
{
    static std::string str() { return "int"; }
};
template <>
struct type_name<unsigned int>
{
    static std::string str() { return "unsigned int"; }
};
template <>
struct type_name<int64_t>
{
    static std::string str() { return "int64_t"; }
};
template <>
struct type_name<uint64_t>
{
    static std::string str() { return "uint64_t"; }
};
template <>
struct type_name<std::string>
{
    static std::string str() { return "string"; }
};
template <typename V>
std::string to_string(V const& v)
{
    std::ostringstream os;
    os << v;
    return os.str();
}
template <typename V>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<V>>;
template <typename V>
using remove_all_t = std::remove_pointer_t<std::remove_cv_t<std::remove_reference_t<V>>>;
template <typename... V>
struct is_integral;
template <typename V>
struct is_integral<V> : public std::integral_constant<bool, std::is_integral<V>::value>
{
};
template <typename V>
struct is_integral<V&> : public std::integral_constant<bool, is_integral<V>::value>
{
};
template <typename V>
struct is_integral<V&&> : public std::integral_constant<bool, is_integral<V>::value>
{
};
template <typename V>
struct is_integral<V const> : public std::integral_constant<bool, is_integral<V>::value>
{
};
template <typename V>
struct is_integral<V const&> : public std::integral_constant<bool, is_integral<V>::value>
{
};

template <typename V, typename U, typename... Others>
struct is_integral<V, U, Others...>
    : public std::integral_constant<bool, std::is_integral<V>::value && is_integral<U, Others...>::value>
{
};
template <typename U>
struct is_pointer : public std::is_pointer<U>
{
};
template <typename U>
struct is_pointer<std::shared_ptr<U>> : public std::true_type
{
};
template <typename U>
struct is_pointer<std::unique_ptr<U>> : public std::true_type
{
};
template <typename U>
inline constexpr bool is_pointer_v = is_pointer<U>::value;

namespace _detail
{

template <typename T>
struct is_complete_helper
{
    template <typename U>
    static auto test(U*) -> std::integral_constant<bool, sizeof(U) == sizeof(U)>;
    static auto test(...) -> std::false_type;
    using type = decltype(test((T*)0));
};
} // namespace _detail
template <typename T>
struct is_complete : public _detail::is_complete_helper<T>::type
{
};

template <class T>
static constexpr bool is_complete_v = is_complete<T>::value;
/**
 *  for Array
 */
/**
 *  std::
 */
template <typename V>
struct rank : public std::integral_constant<std::size_t, std::rank<V>::value>
{
};
template <typename V>
struct rank<V&> : public std::integral_constant<std::size_t, rank<V>::value>
{
};
template <typename V>
struct rank<V&&> : public std::integral_constant<std::size_t, rank<V>::value>
{
};
template <typename V>
struct rank<V const> : public std::integral_constant<std::size_t, rank<V>::value>
{
};
template <typename V>
struct rank<V const&> : public std::integral_constant<std::size_t, rank<V>::value>
{
};

template <typename V>
struct rank<V*> : public std::integral_constant<std::size_t, std::rank<V>::value + 1>
{
};

template <typename V, unsigned _Uint = 0>
struct extent : public std::integral_constant<std::size_t, std::extent<V, _Uint>::value>
{
};
template <typename V>
struct extent<V&> : public std::integral_constant<std::size_t, extent<V>::value>
{
};
template <typename V>
struct extent<V&&> : public std::integral_constant<std::size_t, extent<V>::value>
{
};
template <typename V>
struct extent<V const> : public std::integral_constant<std::size_t, extent<V>::value>
{
};
template <typename V>
struct extent<V const&> : public std::integral_constant<std::size_t, extent<V>::value>
{
};

template <typename V>
struct remove_extent
{
    typedef std::remove_extent_t<V> type;
};
template <typename V>
using remove_extent_t = typename remove_extent<V>::type;
template <typename V>
struct remove_extent<V&>
{
    typedef remove_extent_t<V> type;
};
template <typename V>
struct remove_extent<V&&>
{
    typedef remove_extent_t<V> type;
};
template <typename V>
struct remove_extent<V const&>
{
    typedef remove_extent_t<V> type;
};
template <typename V>
struct remove_extent<V const>
{
    typedef remove_extent_t<V> type;
};

template <typename V>
struct remove_all_extents
{
    typedef std::remove_all_extents_t<V> type;
};
template <typename V>
using remove_all_extents_t = typename remove_all_extents<V>::type;
template <typename V>
struct remove_all_extents<V&>
{
    typedef remove_all_extents_t<V> type;
};
template <typename V>
struct remove_all_extents<V const&>
{
    typedef remove_all_extents_t<V> type;
};
template <typename V>
struct remove_all_extents<V const>
{
    typedef remove_all_extents_t<V> type;
};

template <typename V>
struct rank<std::complex<V>> : public std::integral_constant<std::size_t, 0>
{
};
template <typename V, unsigned int _Uint>
struct extent<std::complex<V>, _Uint> : public std::integral_constant<std::size_t, 1>
{
};
/**
 *  not std::
 */
template <typename V>
struct is_scalar : public std::integral_constant<bool, std::is_arithmetic<V>::value>
{
};
template <typename V>
struct is_scalar<std::complex<V>> : public std::integral_constant<bool, true>
{
};
template <typename V>
struct is_scalar<V&> : public std::integral_constant<bool, is_scalar<V>::value>
{
};
template <typename V>
struct is_scalar<V const> : public std::integral_constant<bool, is_scalar<V>::value>
{
};
template <typename V>
struct is_scalar<V const&> : public std::integral_constant<bool, is_scalar<V>::value>
{
};
template <typename V>
struct is_scalar<V&&> : public std::integral_constant<bool, is_scalar<V>::value>
{
};

template <typename V, std::size_t I = 0, typename SFINAE = void>
struct number_of_elements;
template <typename V, std::size_t I>
struct number_of_elements<V, I, std::enable_if_t<(I >= rank<remove_cvref_t<V>>::value)>>
    : public std::integral_constant<std::size_t, 1>
{
};
template <typename V, std::size_t I>
struct number_of_elements<V, I, std::enable_if_t<(I < rank<remove_cvref_t<V>>::value)>>
    : public std::integral_constant<std::size_t, (extent<remove_cvref_t<V>, I>::value *
                                                  number_of_elements<remove_cvref_t<V>, I + 1>::value)>
{
};

template <typename V, typename SFINAE = void>
struct is_range : public std::integral_constant<bool, false>
{
};

#define HAS_MEMBER_FUNCTION(_FUN_NAME_)                                                                      \
    template <typename _T, typename... _Args>                                                                \
    struct has_member_function_##_FUN_NAME_                                                                  \
    {                                                                                                        \
    private:                                                                                                 \
        typedef std::true_type yes;                                                                          \
        typedef std::false_type no;                                                                          \
                                                                                                             \
        template <typename U>                                                                                \
        static auto test(int) ->                                                                             \
            typename std::enable_if<sizeof...(_Args) == 0, decltype(std::declval<U>()._FUN_NAME_())>::type;  \
                                                                                                             \
        template <typename U>                                                                                \
        static auto test(int) ->                                                                             \
            typename std::enable_if<(sizeof...(_Args) > 0),                                                  \
                                    decltype(std::declval<U>()._FUN_NAME_(std::declval<_Args>()...))>::type; \
                                                                                                             \
        template <typename>                                                                                  \
        static no test(...);                                                                                 \
                                                                                                             \
        typedef decltype(test<_T>(0)) check_result;                                                          \
                                                                                                             \
    public:                                                                                                  \
        static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value;                       \
        typedef check_result type;                                                                           \
    };                                                                                                       \
    template <typename... V>                                                                                 \
    using has_member_function_##_FUN_NAME_##_t = typename has_member_function_##_FUN_NAME_<V...>::type;

//#define HAS_FUNCTION(_FUN_NAME_)                                                 \
//    template <typename... _Args>                                                 \
//    struct has_function_##_FUN_NAME_ {                                           \
//       private:                                                                  \
//        typedef std::true_type yes;                                              \
//        typedef std::false_type no;                                              \
//        template <typename U>                                                    \
//        static auto test(int) -> decltype(_FUN_NAME_(std::declval<_Args>()...)); \
//        static no test(...);                                                     \
//                                                                                 \
//        typedef decltype(test(0)) check_result;                                  \
//                                                                                 \
//       public:                                                                   \
//        static constexpr bool value = !std::is_same<check_result, no>::value;    \
//        typedef check_result type;                                               \
//    };

HAS_MEMBER_FUNCTION(begin);
HAS_MEMBER_FUNCTION(end);

template <typename V>
struct is_range<V, std::enable_if_t<(has_member_function_begin<V>::value && has_member_function_end<V>::value)>>
    : public std::integral_constant<bool, true>
{
};

// template <typename V>
// struct number_of_elements<V, 0> : public std::integral_constant<std::size_t, > {};
// template <typename V>
// struct number_of_elements_help<V, 1> : public std::integral_constant<std::size_t, extent<V>::value> {};
// template <typename V, std::size_t I>i
// struct number_of_elements_help
//    : public std::integral_constant<std::size_t, extent<V, I - 1>::value * number_of_elements_help<V, I -
//    1>::value>
//    {};
// template <typename V>
// struct number_of_elements
//    : public std::integral_constant<std::size_t, number_of_elements_help<V, rank<V>::value>::value> {};
//
// template <typename V, std::size_t I>
// struct number_of_elements<V &, I> : public std::integral_constant<std::size_t, number_of_elements<V, I>::value> {};
// template <typename V, std::size_t I>
// struct number_of_elements<V const, I> : public std::integral_constant<std::size_t, number_of_elements<V, I>::value>
// {};
// template <typename V, std::size_t I>
// struct number_of_elements<V const &, I> : public std::integral_constant<std::size_t, number_of_elements<V, I>::value>
// {
//};
// template <typename V, std::size_t I>
// struct number_of_elements<V &&, I> : public std::integral_constant<std::size_t, number_of_elements<V, I>::value> {};

template <typename V, unsigned int... N>
struct add_extent;
template <typename V, unsigned int... N>
using add_extent_t = typename add_extent<V, N...>::type;
template <typename V, unsigned int N0>
struct add_extent<V, N0>
{
    typedef V type[N0];
};
template <typename V, unsigned int N0, unsigned int N1, unsigned int... N>
struct add_extent<V, N0, N1, N...>
{
    typedef add_extent_t<V, N1, N...> type[N0];
};

template <typename V, typename Src, typename SFINAE = void>
struct copy_extents;
template <typename V, typename Src>
using copy_extents_t = typename copy_extents<V, Src>::type;

template <typename V, typename Src>
struct copy_extents<V, Src, std::enable_if_t<(std::is_reference<Src>::value || std::is_const<Src>::value)>>
{
    typedef copy_extents_t<V, remove_cvref_t<Src>> type;
};
template <typename V, typename Src>
struct copy_extents<V, Src, std::enable_if_t<!(std::is_reference<Src>::value || std::is_const<Src>::value) && (rank<Src>::value == 0)>>
{
    typedef V type;
};
template <typename V, typename Src>
struct copy_extents<
    V, Src, std::enable_if_t<!(std::is_reference<Src>::value || std::is_const<Src>::value) && (rank<Src>::value > 0)>>
{
    typedef copy_extents_t<V, remove_extent_t<Src>> type[extent<Src>::value];
};

template <typename TL, typename TR, typename SFINAE = void>
struct is_similar : public std::integral_constant<bool, false>
{
};
template <typename TL, typename TR>
struct is_similar<TL, TR, std::enable_if_t<std::is_reference<TL>::value || std::is_reference<TR>::value || std::is_const<TL>::value || std::is_const<TR>::value>>
    : public std::integral_constant<bool, is_similar<remove_cvref_t<TL>, remove_cvref_t<TR>>::value>
{
};

template <typename TL, typename TR>
struct is_similar<TL, TR, std::enable_if_t<!(std::is_reference<TL>::value || std::is_reference<TR>::value || std::is_const<TL>::value || std::is_const<TR>::value) && (rank<TL>::value != 0 && rank<TL>::value == rank<TR>::value)>>
    : public std::integral_constant<bool,
                                    (extent<TL>::value == extent<TR>::value) &&
                                        (is_similar<remove_extent_t<TL>, remove_extent_t<TR>>::value)>
{
};
template <typename TL, typename TR>
struct is_similar<TL, TR, std::enable_if_t<!(std::is_reference<TL>::value || std::is_reference<TR>::value || std::is_const<TL>::value || std::is_const<TR>::value) && (rank<TL>::value == 0 && rank<TR>::value == 0)>>
    : public std::integral_constant<bool, true>
{
};

template <typename V, typename SFINAE = void>
struct add_reference
{
    typedef V type;
};
template <typename V>
struct add_reference<V, std::enable_if_t<std::is_array<V>::value>>
{
    typedef V const& type;
};
template <typename V>
using add_reference_t = typename add_reference<V>::type;

template <typename V, std::size_t N>
struct nested_initializer_list;

template <typename V, std::size_t N>
using nested_initializer_list_t = typename nested_initializer_list<V, N>::type;

template <typename V>
struct nested_initializer_list<V, 0>
{
    typedef V type;
};

template <typename V>
struct nested_initializer_list<V, 1>
{
    typedef std::initializer_list<V> type;
};

template <typename V, std::size_t N>
struct nested_initializer_list
{
    typedef std::initializer_list<nested_initializer_list_t<V, N - 1>> type;
};
template <typename V, unsigned int N>
using make_nested_initializer_list = typename nested_initializer_list<V, N>::type;

template <typename V>
void nested_initializer_list_traits_dims_helper(std::size_t* dims, V const& v){
    //    dims[0] = 1;
};
template <typename V>
void nested_initializer_list_traits_dims_helper(std::size_t* dims, std::initializer_list<V> const& list)
{
    dims[0] = std::max(dims[0], list.size());
    for (auto const& item : list)
    {
        nested_initializer_list_traits_dims_helper(dims + 1, item);
    }
};
template <typename V, unsigned int N>
void nested_initializer_list_traits_dims(std::size_t dims[N], make_nested_initializer_list<V, N> const& list)
{
    for (int i = 0; i < N; ++i)
    {
        dims[i] = 0;
    }
    nested_initializer_list_traits_dims_helper(dims, list);
};

template <typename V>
struct number_of_dimensions : public std::integral_constant<unsigned int, 0>
{
};
template <typename V>
struct number_of_dimensions<V&> : public std::integral_constant<unsigned int, number_of_dimensions<V>::value>
{
};
template <typename V>
struct number_of_dimensions<V&&> : public std::integral_constant<unsigned int, number_of_dimensions<V>::value>
{
};
template <typename V>
struct number_of_dimensions<V const> : public std::integral_constant<unsigned int, number_of_dimensions<V>::value>
{
};
template <typename V>
struct number_of_dimensions<V const&> : public std::integral_constant<unsigned int, number_of_dimensions<V>::value>
{
};
template <typename V>
struct number_of_dimensions<V*> : public std::integral_constant<unsigned int, number_of_dimensions<V>::value + 1>
{
};

template <typename V>
struct remove_all_dimensions
{
    typedef V type;
};
template <typename V>
struct remove_all_dimensions<V&>
{
    typedef typename remove_all_dimensions<V>::type type;
};
template <typename V>
struct remove_all_dimensions<V&&>
{
    typedef typename remove_all_dimensions<V>::type type;
};
template <typename V>
struct remove_all_dimensions<V const>
{
    typedef typename remove_all_dimensions<V>::type type;
};
template <typename V>
struct remove_all_dimensions<V const&>
{
    typedef typename remove_all_dimensions<V>::type type;
};
template <typename V>
struct remove_all_dimensions<V*>
{
    typedef std::remove_pointer_t<V> type;
};
template <typename V>
using remove_all_dimensions_t = typename remove_all_dimensions<V>::type;

// template <typename V, unsigned int N = 1>
// struct remove_dimension {
//    typedef V type;
//};
//
// template <typename V, unsigned int N>
// struct remove_dimension<V &, N> {
//    typedef typename remove_dimension<V, N>::type type;
//};
// template <typename V, unsigned int N>
// struct remove_dimension<V &&, N> {
//    typedef typename remove_dimension<V, N>::type type;
//};
// template <typename V, unsigned int N>
// struct remove_dimension<V const, N> {
//    typedef typename remove_dimension<V, N>::type type;
//};
// template <typename V, unsigned int N>
// struct remove_dimension<V const &, N> {
//    typedef typename remove_dimension<V, N>::type type;
//};
// template <typename V>
// struct remove_dimension<V *, 0> {
//    typedef V type;
//};
// template <typename V, unsigned int N>
// struct remove_dimension<V *, N> {
//    typedef typename remove_dimension<V, N - 1>::type type;
//};
// template <typename V, unsigned int N = 1>
// using remove_dimension_t = typename remove_dimension<V, N>::type;
//
// template <typename V, unsigned int N = 1>
// struct add_dimension;
// template <typename V>
// struct add_dimension<V, 0> {
//    typedef V type;
//};
// template <typename V>
// struct add_dimension<V, 1> {
//    typedef V *type;
//};
// template <typename V, unsigned int N>
// struct add_dimension {
//    typedef typename add_dimension<V, N - 1>::type *type;
//};
// template <typename V, unsigned int N>
// struct add_dimension<V &, N> {
//    typedef typename add_dimension<V, N>::type type;
//};
// template <typename V, unsigned int N>
// struct add_dimension<V &&, N> {
//    typedef typename add_dimension<V, N>::type type;
//};
// template <typename V, unsigned int N>
// struct add_dimension<V const, N> {
//    typedef typename add_dimension<V, N>::type type;
//};
// template <typename V, unsigned int N>
// struct add_dimension<V const &, N> {
//    typedef typename add_dimension<V, N>::type type;
//};
// template <typename V, unsigned int N = 1>
// using add_dimension_t = typename add_dimension<V, N>::type;
//
// template <typename U>
// struct nested_initializer_list_traits {
//    static constexpr int number_of_levels = 0;
//    static void GetDims(U const &list, int *dims) {}
//};
// template <typename U>
// struct nested_initializer_list_traits<std::initializer_list<U>> {
//    static constexpr int number_of_levels = nested_initializer_list_traits<U>::number_of_levels + 1;
//    static void GetDims(std::initializer_list<U> const &list, std::size_t *dims) { dims[0] = list.size(); }
//};
// template <typename U>
// struct nested_initializer_list_traits<std::initializer_list<std::initializer_list<U>>> {
//    static constexpr int number_of_levels = nested_initializer_list_traits<U>::number_of_levels + 2;
//    static void GetDims(std::initializer_list<std::initializer_list<U>> const &list, std::size_t *dims) {
//        dims[0] = list.size();
//        std::size_t max_length = 0;
//        for (auto const &item : list) { max_length = (max_length < item.size()) ? item.size() : max_length; }
//        dims[1] = max_length;
//    }
//};
//
// template <int... I>
// struct assign_nested_initializer_list;
//
// template <>
// struct assign_nested_initializer_list<> {
//    template <typename U, typename TR>
//    __host__ __device__ static void apply(U &u, TR const &rhs) {
//        u = rhs;
//    }
//};
//
// template <int I0, int... I>
// struct assign_nested_initializer_list<I0, I...> {
//    template <typename U, typename TR>
//    __host__ __device__ static void apply(U &u, std::initializer_list<TR> const &rhs) {
//        static_assert(is_indexable<U, int>::value, " illegal value_type_info");
//
//        auto it = rhs.begin();
//        auto ie = rhs.end();
//
//        for (int i = 0; i < I0 && it != ie; ++i, ++it) { assign_nested_initializer_list<I...>::apply(u[i],
//        *it); }
//    }
//};
template <typename _Tp, _Tp... I>
struct min;
template <typename _Tp, _Tp I0>
struct min<_Tp, I0> : public std::integral_constant<_Tp, I0>
{
};
template <typename _Tp, _Tp I0, _Tp I1>
struct min<_Tp, I0, I1> : public std::integral_constant<_Tp, (I0 < I1 ? I0 : I1)>
{
};
template <typename _Tp, _Tp I0, _Tp I1, _Tp... I>
struct min<_Tp, I0, I1, I...> : public std::integral_constant<_Tp, min<_Tp, I0, min<_Tp, I1, I...>::value>::value>
{
};

template <typename _Tp, _Tp... I>
struct max;
template <typename _Tp, _Tp I0>
struct max<_Tp, I0> : public std::integral_constant<_Tp, I0>
{
};
template <typename _Tp, _Tp I0, _Tp I1>
struct max<_Tp, I0, I1> : public std::integral_constant<_Tp, (I0 > I1 ? I0 : I1)>
{
};
template <typename _Tp, _Tp I0, _Tp I1, _Tp... I>
struct max<_Tp, I0, I1, I...> : public std::integral_constant<_Tp, max<_Tp, I0, max<_Tp, I1, I...>::value>::value>
{
};

template <typename _Tp, _Tp... I>
struct sum;
template <typename _Tp, _Tp I0>
struct sum<_Tp, I0> : public std::integral_constant<_Tp, I0>
{
};
template <typename _Tp, _Tp I0, _Tp I1>
struct sum<_Tp, I0, I1> : public std::integral_constant<_Tp, I0 + I1>
{
};
template <typename _Tp, _Tp I0, _Tp I1, _Tp... I>
struct sum<_Tp, I0, I1, I...> : public std::integral_constant<_Tp, I0 + sum<_Tp, I1, I...>::value>
{
};

template <typename _Tp, _Tp... I>
struct product;
template <typename _Tp, _Tp I0>
struct product<_Tp, I0> : public std::integral_constant<_Tp, I0>
{
};
template <typename _Tp, _Tp I0, _Tp I1>
struct product<_Tp, I0, I1> : public std::integral_constant<_Tp, I0 * I1>
{
};
template <typename _Tp, _Tp I0, _Tp I1, _Tp... I>
struct product<_Tp, I0, I1, I...> : public std::integral_constant<_Tp, I0 * sum<_Tp, I1, I...>::value>
{
};

//**********************************************************************************************************************

namespace _detail
{

template <typename _TFun, typename... _Args>
struct check_invocable
{
private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> decltype(std::declval<_U>()(std::declval<_Args>()...));

    template <typename>
    static no test(...);

public:
    typedef decltype(test<_TFun>(0)) type;

    static constexpr bool value = !std::is_same<type, no>::value;
};

template <typename _TFun, typename _Arg>
struct check_indexable
{
private:
    typedef std::true_type yes;
    typedef std::false_type no;

    template <typename _U>
    static auto test(int) -> decltype(std::declval<_U>()[std::declval<_Arg>()]);

    template <typename>
    static no test(...);

public:
    typedef decltype(test<_TFun>(0)) type;

    static constexpr bool value = !std::is_same<type, no>::value;
};
} // namespace _detail

template <typename TFun, typename... Args>
struct invoke_result
{
    typedef typename _detail::check_invocable<TFun, Args...>::type type;
};

template <typename TFun, typename... Args>
using invoke_result_t = typename invoke_result<TFun, Args...>::type;

template <typename TFun, typename... Args>
struct is_invocable
    : public std::integral_constant<
          bool, !std::is_same<typename _detail::check_invocable<TFun, Args...>::type, std::false_type>::value>
{
};

template <typename R, typename TFun, typename... Args>
struct is_invocable_r
    : public std::integral_constant<bool,
                                    std::is_same<typename _detail::check_invocable<TFun, Args...>::type, R>::value>
{
};

template <typename U, typename ArgsTypelist, typename Enable = void>
struct InvokeHelper_
{
    template <typename V, typename... Args>
    static decltype(auto) eval(V& v, Args&&... args)
    {
        return v;
    }
};

template <typename U, typename... Args>
struct InvokeHelper_<U, std::tuple<Args...>, std::enable_if_t<is_invocable<U, Args...>::value>>
{
    template <typename V, typename... Args2>
    static decltype(auto) eval(V& v, Args2&&... args)
    {
        return v(std::forward<Args>(args)...);
    }
};
template <typename U, typename... Args>
struct InvokeHelper_<const U, std::tuple<Args...>, std::enable_if_t<is_invocable<const U, Args...>::value>>
{
    template <typename V, typename... Args2>
    static decltype(auto) eval(V const& v, Args2&&... args)
    {
        return v(std::forward<Args>(args)...);
    }
};
template <typename U, typename... Args>
decltype(auto) invoke(U& v, Args&&... args)
{
    return InvokeHelper_<U, std::tuple<Args...>>::eval(v, std::forward<Args>(args)...);
}
template <typename U, typename... Args>
decltype(auto) invoke(U const& v, Args&&... args)
{
    return InvokeHelper_<U, std::tuple<Args...>>::eval(v, std::forward<Args>(args)...);
}
//**********************************************************************************************************************
/**
* @ref http://en.cppreference.com/w/cpp/types/remove_extent
* If T is '''is_indexable''' by '''S''', provides the member typedef type equal to
* decltyp(T[S])
* otherwise type is T. Note that if T is a multidimensional array, only the first dimension is
* removed.
*/

template <typename T, typename TI = int>
struct is_indexable : public std::integral_constant<
                          bool, !std::is_same<typename _detail::check_indexable<T, TI>::type, std::false_type>::value>
{
};
//**********************************************************************************************************************

template <typename U, typename V, typename Enable = void>
struct is_iterator_of
{
    static const bool value = false;
};

template <typename U, typename V>
struct is_iterator_of<U, V, std::enable_if_t<std::is_same_v<U, typename std::iterator_traits<V>::value_type> || //
                                             std::is_same_v<std::remove_const_t<U>, typename std::iterator_traits<V>::value_type>>>
{
    static const bool value = true;
};

template <typename U, typename V>
static const bool is_iterator_of_v = is_iterator_of<U, V>::value;

//**********************************************************************************************************************

template <typename... T>
struct regisiter_type_tag;

template <typename _Head>
struct regisiter_type_tag<_Head>
{
    typedef _Head tags;
};

template <typename _Head, typename T>
struct regisiter_type_tag<_Head, T>
{
    struct tags : public _Head
    {
        enum
        {
            UNNAMED = _Head::_LAST_PLACE_HOLDER,
            _LAST_PLACE_HOLDER
        };
    };
};

template <typename _Head, typename First, typename... Others>
struct regisiter_type_tag<_Head, First, Others...>
{
    typedef typename regisiter_type_tag<typename regisiter_type_tag<_Head, First>::tags, Others...>::tags tags;
};

/**
 *  TODO: (salmon 2020.07.29): check duplicate type 
 * 
*/
namespace _detail
{
struct _tags_head
{
    enum
    {
        _LAST_PLACE_HOLDER
    };
};
} // namespace _detail
template <typename... T>
struct type_tags_traits
{
    typedef typename regisiter_type_tag<_detail::_tags_head, T...>::tags tags;
};

namespace _detail
{
template <typename...>
struct _type_tags;

template <typename... TypeLists>
struct _type_tags<std::variant<TypeLists...>>
{
    typedef typename type_tags_traits<TypeLists...>::tags tags;
};
} // namespace _detail

template <typename V>
using type_tags = typename _detail::_type_tags<V>::tags;

//**********************************************************************************************************************

template <typename T>
struct is_variant
{
    static const bool value = false;
};

template <typename... T>
struct is_variant<std::variant<T...>>
{
    static const bool value = true;
};

// template <typename... First>
// struct concatenate<std::variant<First...>>
// {
//     typedef std::variant<First...> type;
// };

namespace _detail
{
template <typename Frist, typename Second, typename Enable = void>
struct concatenate_impl;

template <typename... First, typename... Second>
struct concatenate_impl<std::variant<First...>, std::variant<Second...>>
{
    typedef std::variant<First..., Second...> type;
};

template <typename First, typename Second>
struct concatenate_impl<First, Second, std::enable_if_t<!is_variant<First>::value && !is_variant<Second>::value>>
{
    typedef std::variant<First, Second> type;
};

template <typename First, typename... Second>
struct concatenate_impl<First, std::variant<Second...>, std::enable_if_t<!is_variant<First>::value>>
{
    typedef std::variant<First, Second...> type;
};

template <typename... First, typename Second>
struct concatenate_impl<std::variant<First...>, Second, std::enable_if_t<!is_variant<Second>::value>>
{
    typedef std::variant<First..., Second> type;
};
} // namespace _detail

template <typename...>
struct concatenate;

template <typename First>
struct concatenate<First>
{
    typedef First type;
};
template <typename First, typename Second, typename... Others>
struct concatenate<First, Second, Others...>
{
    typedef typename concatenate<typename _detail::concatenate_impl<First, Second>::type, Others...>::type type;
};

template <typename... TypeList>
using concatenate_t = typename concatenate<TypeList...>::type;

namespace _detail
{
template <class T, T...>
struct make_integer_sequence_impl;

template <class T, T BEGIN, T END, T STEP>
struct make_integer_sequence_impl<T, BEGIN, END, STEP>
{

    template <typename V>
    struct _idx_map;

    template <T... I>
    struct _idx_map<std::integer_sequence<T, I...>>
    {
        typedef std::index_sequence<(I * STEP + BEGIN)...> type;
    };

    typedef typename _idx_map<std::make_integer_sequence<T, (END - BEGIN) / STEP>>::type type;
};
template <class T, T N>
struct make_integer_sequence_impl<T, N>
{
    typedef typename make_integer_sequence_impl<T, 0, N, 1>::type type;
};
template <class T, T B, T E>
struct make_integer_sequence_impl<T, B, E>
{
    typedef typename make_integer_sequence_impl<T, B, E, 1>::type type;
};
} // namespace _detail
template <class T, T... I>
using make_integer_sequence = typename _detail::make_integer_sequence_impl<T, I...>::type;

template <std::size_t... I>
using make_index_sequence = make_integer_sequence<std::size_t, I...>;

template <typename...>
struct select_variant;

template <typename... T, std::size_t... I>
struct select_variant<std::variant<T...>, std::index_sequence<I...>>
{
    typedef std::variant<std::variant_alternative_t<I, std::variant<T...>>...> type;
};

template <typename T, std::size_t... I>
using select_variant_t = typename select_variant<T, make_index_sequence<I...>>::type;

namespace _detail
{
template <template <typename...> class T0, typename TypeLists>
struct template_copy_type_args_impl;
template <template <typename...> class T0, template <typename...> class T1, typename... Types>
struct template_copy_type_args_impl<T0, T1<Types...>>
{
    typedef T0<Types...> type;
};
} // namespace _detail

template <template <typename...> class T0, typename T1>
using template_copy_type_args = typename _detail::template_copy_type_args_impl<T0, T1>::type;

template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...)->overloaded<Ts...>;

} // namespace traits
} // namespace sp

#define M_REGISITER_TYPE_TAG(_TAG_, ...)           \
    namespace sp                                   \
    {                                              \
    namespace traits                               \
    {                                              \
    template <typename _Head>                      \
    struct regisiter_type_tag<_Head, __VA_ARGS__>  \
    {                                              \
        struct tags : public _Head                 \
        {                                          \
            enum                                   \
            {                                      \
                _TAG_ = _Head::_LAST_PLACE_HOLDER, \
                _LAST_PLACE_HOLDER                 \
            };                                     \
        };                                         \
    };                                             \
    }                                              \
    }

M_REGISITER_TYPE_TAG(Empty, std::nullptr_t);                                              //Empty,
M_REGISITER_TYPE_TAG(Block, std::tuple<std::shared_ptr<void>, int, std::vector<size_t>>); //Block
M_REGISITER_TYPE_TAG(String, std::string);                                                //String,
M_REGISITER_TYPE_TAG(Bool, bool);                                                         //Boolean,
M_REGISITER_TYPE_TAG(Int, int);                                                           //Integer,
M_REGISITER_TYPE_TAG(Long, long);                                                         //Long,
M_REGISITER_TYPE_TAG(Float, float);                                                       //Float,
M_REGISITER_TYPE_TAG(Double, double);                                                     //Double,
M_REGISITER_TYPE_TAG(Complex, std::complex<double>);                                      //Complex,
M_REGISITER_TYPE_TAG(IntVec3, std::array<int, 3>);                                        //IntVec3,
M_REGISITER_TYPE_TAG(LongVec3, std::array<long, 3>);                                      //LongVec3,
M_REGISITER_TYPE_TAG(FloatVec3, std::array<float, 3>);                                    //FloatVec3,
M_REGISITER_TYPE_TAG(DoubleVec3, std::array<double, 3>);                                  //DoubleVec3,
M_REGISITER_TYPE_TAG(ComplexVec3, std::array<std::complex<double>, 3>);                   //ComplexVec3,
M_REGISITER_TYPE_TAG(UNKNOWN, std::any);                                                  //Other

#define M_REGISITER_TYPE_TAG_TEMPLATE(_TAG_, _TMPL_)         \
    namespace sp                                             \
    {                                                        \
    namespace traits                                         \
    {                                                        \
    template <typename _Head, typename... _PARAMETERS>       \
    struct regisiter_type_tag<_Head, _TMPL_<_PARAMETERS...>> \
    {                                                        \
        struct tags : public _Head                           \
        {                                                    \
            enum                                             \
            {                                                \
                _TAG_ = _Head::_LAST_PLACE_HOLDER,           \
                _LAST_PLACE_HOLDER                           \
            };                                               \
        };                                                   \
    };                                                       \
    }                                                        \
    }

#endif // SP_TYPETRAITS_H_
