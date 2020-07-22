//
// Created by salmon on 17-4-28.
//

#ifndef SIMPLA_EXPRESSIONTEMPLATE_H
#define SIMPLA_EXPRESSIONTEMPLATE_H

#include "utility/TypeTraits.h"
#include <cmath>
#include <complex>
#include <tuple>
#include <utility>
#define __host__
#define __device__
#define ENABLE_IF(_COND_) std::enable_if_t<_COND_, void> *_p = nullptr

namespace sp
{
    template <typename...>
    class Expression;

    namespace traits
    {

        template <typename TExpr>
        struct is_expression : public std::false_type
        {
        };
        template <typename... T>
        struct is_expression<Expression<T...>> : public std::true_type
        {
        };
        template <typename... T>
        struct is_expression<Expression<T...> &> : public std::true_type
        {
        };
        template <typename... T>
        struct is_expression<Expression<T...> &&> : public std::true_type
        {
        };
        template <typename... T>
        struct is_expression<Expression<T...> const &> : public std::true_type
        {
        };
        template <typename... T>
        struct is_expression<Expression<T...> const> : public std::true_type
        {
        };
        template <typename V>
        struct is_complex : public std::false_type
        {
        };
        template <typename V>
        struct is_complex<std::complex<V>> : public std::true_type
        {
        };
        template <typename V>
        struct is_complex<std::complex<V> &> : public std::true_type
        {
        };
        template <typename V>
        struct is_complex<std::complex<V> &&> : public std::true_type
        {
        };
        template <typename V>
        struct is_complex<std::complex<V> const> : public std::true_type
        {
        };
        template <typename V>
        struct is_complex<std::complex<V> const &> : public std::true_type
        {
        };

        template <typename TOP, typename... Args>
        struct rank<Expression<TOP, Args...>>
            : public std::integral_constant<std::size_t, max<std::size_t, rank<Args>::value...>::value>
        {
        };
        template <typename TOP, typename... Args, unsigned int I>
        struct extent<Expression<TOP, Args...>, I>
            : public std::integral_constant<std::size_t, min<std::size_t, extent<Args, I>::value...>::value>
        {
        };

        template <typename TOP, typename... Args>
        struct number_of_dimensions<Expression<TOP, Args...>>
            : public std::integral_constant<std::size_t, min<std::size_t, number_of_dimensions<Args>::value...>::value>
        {
        };

    } // namespace traits
    namespace calculus
    {

        template <typename REDUCTION, typename TExpr, typename SFINAE = void>
        struct ReduceExpressionHelper;
        template <typename REDUCTION, typename TExpr>
        auto reduce_expression(TExpr const &expr)
        {
            return ReduceExpressionHelper<REDUCTION, traits::remove_cvref_t<TExpr>>::eval(expr);
        };
        template <typename REDUCTION, typename TExpr>
        struct ReduceExpressionHelper<REDUCTION, TExpr, std::enable_if_t<(traits::rank<TExpr>::value == 0 && traits::number_of_dimensions<TExpr>::value == 0)>>
        {
            static auto eval(TExpr const &expr) { return expr; }
        };

        template <typename TRes, typename TExpr, typename SFINAE = void>
        struct EvaluateExpressionHelper;

        template <typename TRes, typename TExpr>
        int evaluate_expression(TRes &&res, TExpr const &expr)
        {
            return EvaluateExpressionHelper<TRes, TExpr>::eval(std::forward<TRes>(res), expr);
        };
        template <typename TRes>
        int evaluate_expression(TRes &&res, std::nullptr_t)
        {
            return 1;
        };
        // template <typename TRes, typename TExpr>
        // auto evaluate_expression(TRes &&res, TExpr const &expr) {
        //    return EvaluateExpressionHelper<TRes, TExpr>::eval(std::forward<TRes>(res), expr);
        //};
        template <typename TRes, typename TExpr>
        struct EvaluateExpressionHelper<
            TRes, TExpr, std::enable_if_t<(traits::rank<TRes>::value == 0 && traits::number_of_dimensions<TRes>::value == 0)>>
        {
            template <typename R, typename E>
            static int eval(R &&lhs, E const &rhs)
            {
                lhs = rhs;
                return 1;
            }
        };

        template <typename V, typename SFINAE = void>
        struct RecursiveGetHelper
        {
            template <std::size_t I, typename U>
            static decltype(auto) eval(U &&v)
            {
                return v;
            }
        };

        template <std::size_t I, typename TExpr>
        decltype(auto) get_r(TExpr &&expr)
        {
            return RecursiveGetHelper<traits::remove_cvref_t<TExpr>>::template eval<I>(std::forward<TExpr>(expr));
        }

        template <typename... V>
        struct RecursiveGetHelper<std::tuple<V...>>
        {
            template <std::size_t I, typename U>
            static decltype(auto) eval(U &&u)
            {
                return std::get<I>(std::forward<U>(u));
            }
        };

        template <typename TOP, typename... Args>
        struct RecursiveGetHelper<Expression<TOP, Args...>>
        {
            typedef Expression<TOP, Args...> expr_type;

            template <std::size_t I, typename U, std::size_t... IDX>
            static decltype(auto) helper(U &&expr, std::index_sequence<IDX...>)
            {
                return expr.m_op_(get_r<I>(std::get<IDX>(expr.m_args_))...);
            }

            template <std::size_t I, typename U>
            static decltype(auto) eval(U &&expr)
            {
                return helper<I>(std::forward<U>(expr), std::index_sequence_for<Args...>());
            }
        };

        template <typename REDUCTION, typename TExpr, typename SFINAE>
        struct ReduceExpressionHelper;

        template <typename REDUCTION, typename TExpr>
        struct ReduceExpressionHelper<REDUCTION, TExpr, std::enable_if_t<(traits::rank<TExpr>::value > 0)>>
        {
            template <typename First>
            static auto reduce(First const &first)
            {
                return first;
            }
            template <typename First, typename Second>
            static auto reduce(First const &first, Second const &second)
            {
                return REDUCTION::eval(first, second);
            }
            template <typename First, typename Second, typename... Others>
            static auto reduce(First const &first, Second const &second, Others &&... others)
            {
                return REDUCTION::eval(first, reduce(second, std::forward<Others>(others)...));
            }
            template <std::size_t... IDX>
            static auto eval_helper(TExpr const &expr, std::integer_sequence<std::size_t, IDX...>)
            {
                return reduce(get_r<IDX>(expr)...);
            }
            static auto eval(TExpr const &expr)
            {
                return eval_helper(expr, std::make_index_sequence<traits::number_of_elements<TExpr>::value>());
            }
        };

        template <typename TRes, typename TExpr>
        struct EvaluateExpressionHelper<TRes, TExpr, std::enable_if_t<(traits::rank<TRes>::value > 0)>>
        {
            template <std::size_t... IDX>
            static int eval_helper(TRes &&lhs, TExpr const &rhs, std::integer_sequence<std::size_t, IDX...>)
            {
                return utility::sum_n(evaluate_expression(get_r<IDX>(std::forward<TRes>(lhs)), get_r<IDX>(rhs))...);
            }
            static int eval(TRes &&lhs, TExpr const &rhs)
            {
                return eval_helper(std::forward<TRes>(lhs), rhs,
                                   std::make_index_sequence<traits::number_of_elements<TRes>::value>());
            }
        };
        template <typename TRes, typename U>
        struct EvaluateExpressionHelper<TRes, std::initializer_list<U>, std::enable_if_t<(traits::rank<TRes>::value > 0)>>
        {
            template <typename LHS, typename RHS>
            static int eval(LHS &&lhs, RHS const &rhs)
            {
                static constexpr traits::remove_all_extents_t<traits::remove_cvref_t<LHS>> snan =
                    std::numeric_limits<traits::remove_all_extents_t<traits::remove_cvref_t<LHS>>>::signaling_NaN();
                int i = 0;
                auto it = std::begin(rhs);
                auto ie = std::end(rhs);
                for (int i = 0; i < traits::extent<TRes>::value; ++i)
                {
                    if (it == ie)
                    {
                        evaluate_expression(lhs[i], snan);
                    }
                    else
                    {
                        evaluate_expression(lhs[i], *it);
                    }
                    ++it;
                }

                return 1;
            }
        };
        // template <typename Dest, typename Src>
        // void Copy(Dest &dest, Src const &src, std::enable_if_t<(traits::rank<Dest>::value == 0)> *_p = nullptr) {
        //    dest = src;
        //};
        // template <typename Dest, typename Src>
        // void Copy(Dest &dest, Src const &src, std::enable_if_t<(traits::rank<Dest>::value > 0)> *_p = nullptr) {
        //    std::size_t m_count_ = 0;
        //    for (auto const &item : src) {
        //        Copy(dest[m_count_], item);
        //        ++m_count_;
        //        if (m_count_ >= traits::extent<Dest>::value) { break; }
        //    }
        //};
    } // namespace calculus

    template <typename TOP, typename... Args>
    struct Expression<TOP, Args...>
    {
        typedef Expression<TOP, Args...> this_type;
        std::tuple<sp::traits::add_reference_t<Args>...> m_args_;
        TOP m_op_;
        __host__ __device__ Expression(this_type const &that) : m_args_(that.m_args_) {}
        __host__ __device__ Expression(this_type &&that) noexcept : m_args_(std::move(that.m_args_)) {}
        template <typename... U>
        __host__ __device__ explicit Expression(U &&... args) : m_args_(std::forward<U>(args)...) {}
        __host__ __device__ virtual ~Expression() = default;
        template <typename T>
        __host__ __device__ operator T() const
        {
            T res;
            calculus::evaluate_expression(res, *this);
            return std::move(res);
        }
    };

    namespace tags
    {
#define _SP_DEFINE_TAG_BINARY_OPERATOR(_TAG_, _OP_)                                   \
    struct _TAG_                                                                      \
    {                                                                                 \
        template <typename TL, typename TR>                                           \
        __host__ __device__ static constexpr auto eval(TL const &l, TR const &r)      \
        {                                                                             \
            return ((l _OP_ r));                                                      \
        }                                                                             \
        template <typename TL, typename TR>                                           \
        __host__ __device__ constexpr auto operator()(TL const &l, TR const &r) const \
        {                                                                             \
            return ((l _OP_ r));                                                      \
        }                                                                             \
    };

#define _SP_DEFINE_TAG_UNARY_OPERATOR(_TAG_, _OP_)                       \
    struct _TAG_                                                         \
    {                                                                    \
        template <typename TL>                                           \
        __host__ __device__ static constexpr auto eval(TL const &l)      \
        {                                                                \
            return (_OP_(l));                                            \
        }                                                                \
        template <typename TL>                                           \
        __host__ __device__ constexpr auto operator()(TL const &l) const \
        {                                                                \
            return _OP_(l);                                              \
        }                                                                \
    };

#define _SP_DEFINE_TAG_BINARY_FUNCTION(_TAG_, _FUN_)                                  \
    struct _TAG_                                                                      \
    {                                                                                 \
        template <typename TL, typename TR>                                           \
        __host__ __device__ static constexpr auto eval(TL const &l, TR const &r)      \
        {                                                                             \
            return (_FUN_(l, r));                                                     \
        }                                                                             \
        template <typename TL, typename TR>                                           \
        __host__ __device__ constexpr auto operator()(TL const &l, TR const &r) const \
        {                                                                             \
            return (_FUN_(l, r));                                                     \
        }                                                                             \
    };

#define _SP_DEFINE_TAG_UNARY_FUNCTION(_TAG_, _FUN_)                      \
    struct _TAG_                                                         \
    {                                                                    \
        template <typename TL>                                           \
        __host__ __device__ static constexpr auto eval(TL const &l)      \
        {                                                                \
            return (_FUN_(l));                                           \
        }                                                                \
        template <typename TL>                                           \
        __host__ __device__ constexpr auto operator()(TL const &l) const \
        {                                                                \
            return _FUN_(l);                                             \
        }                                                                \
    };

        _SP_DEFINE_TAG_BINARY_OPERATOR(addition, +)
        _SP_DEFINE_TAG_BINARY_OPERATOR(subtraction, -)
        _SP_DEFINE_TAG_BINARY_OPERATOR(multiplication, *)
        _SP_DEFINE_TAG_BINARY_OPERATOR(division, /)
        _SP_DEFINE_TAG_BINARY_OPERATOR(modulo, %)

        _SP_DEFINE_TAG_UNARY_OPERATOR(bitwise_not, ~)
        _SP_DEFINE_TAG_BINARY_OPERATOR(bitwise_xor, ^)
        _SP_DEFINE_TAG_BINARY_OPERATOR(bitwise_and, &)
        _SP_DEFINE_TAG_BINARY_OPERATOR(bitwise_or, |)
        _SP_DEFINE_TAG_BINARY_OPERATOR(bitwise_left_shift, <<)
        _SP_DEFINE_TAG_BINARY_OPERATOR(bitwise_right_shifit, >>)

        _SP_DEFINE_TAG_UNARY_OPERATOR(unary_plus, +)
        _SP_DEFINE_TAG_UNARY_OPERATOR(unary_minus, -)

        _SP_DEFINE_TAG_UNARY_OPERATOR(logical_not, !)
        _SP_DEFINE_TAG_BINARY_OPERATOR(logical_and, &&)
        _SP_DEFINE_TAG_BINARY_OPERATOR(logical_or, ||)

        _SP_DEFINE_TAG_BINARY_OPERATOR(not_equal_to, !=)
        _SP_DEFINE_TAG_BINARY_OPERATOR(equal_to, ==)
        _SP_DEFINE_TAG_BINARY_OPERATOR(less, <)
        _SP_DEFINE_TAG_BINARY_OPERATOR(greater, >)
        _SP_DEFINE_TAG_BINARY_OPERATOR(less_equal, <=)
        _SP_DEFINE_TAG_BINARY_OPERATOR(greater_equal, >=)

        _SP_DEFINE_TAG_UNARY_FUNCTION(cos, std::cos)
        _SP_DEFINE_TAG_UNARY_FUNCTION(acos, std::acos)
        _SP_DEFINE_TAG_UNARY_FUNCTION(cosh, std::cosh)
        _SP_DEFINE_TAG_UNARY_FUNCTION(sin, std::sin)
        _SP_DEFINE_TAG_UNARY_FUNCTION(asin, std::asin)
        _SP_DEFINE_TAG_UNARY_FUNCTION(sinh, std::sinh)
        _SP_DEFINE_TAG_UNARY_FUNCTION(tan, std::tan)
        _SP_DEFINE_TAG_UNARY_FUNCTION(tanh, std::tanh)
        _SP_DEFINE_TAG_UNARY_FUNCTION(atan, std::atan)
        _SP_DEFINE_TAG_UNARY_FUNCTION(exp, std::exp)
        _SP_DEFINE_TAG_UNARY_FUNCTION(log, std::log)
        _SP_DEFINE_TAG_UNARY_FUNCTION(log10, std::log10)
        _SP_DEFINE_TAG_UNARY_FUNCTION(sqrt, std::sqrt)
        _SP_DEFINE_TAG_BINARY_FUNCTION(atan2, std::atan2)
        _SP_DEFINE_TAG_BINARY_FUNCTION(pow, std::pow)

#undef _SP_DEFINE_TAG_BINARY_OPERATOR
#undef _SP_DEFINE_TAG_UNARY_OPERATOR
#undef _SP_DEFINE_TAG_BINARY_FUNCTION
#undef _SP_DEFINE_TAG_UNARY_FUNCTION

    } // namespace tags

#define _SP_DEFINE_BINARY_FUNCTION(_TAG_, _FUN_)                                                   \
    template <typename TL, typename TR>                                                            \
    __host__ __device__ auto _FUN_(TL const &lhs, TR const &rhs)                                   \
        ->std::enable_if_t<(traits::is_expression<TL>::value || traits::is_expression<TR>::value), \
                           Expression<tags::_TAG_, TL, TR>>                                        \
    {                                                                                              \
        return Expression<tags::_TAG_, TL, TR>(lhs, rhs);                                          \
    };

#define _SP_DEFINE_UNARY_FUNCTION(_TAG_, _FUN_)                                             \
    template <typename TL>                                                                  \
    __host__ __device__ auto _FUN_(TL const &lhs)                                           \
        ->std::enable_if_t<(traits::is_expression<TL>::value), Expression<tags::_TAG_, TL>> \
    {                                                                                       \
        return Expression<tags::_TAG_, TL>(lhs);                                            \
    }

    _SP_DEFINE_BINARY_FUNCTION(addition, operator+)
    _SP_DEFINE_BINARY_FUNCTION(subtraction, operator-)
    _SP_DEFINE_BINARY_FUNCTION(multiplication, operator*)
    _SP_DEFINE_BINARY_FUNCTION(division, operator/)
    _SP_DEFINE_BINARY_FUNCTION(modulo, operator%)

    _SP_DEFINE_UNARY_FUNCTION(bitwise_not, operator~)
    _SP_DEFINE_BINARY_FUNCTION(bitwise_xor, operator^)
    _SP_DEFINE_BINARY_FUNCTION(bitwise_and, operator&)
    _SP_DEFINE_BINARY_FUNCTION(bitwise_or, operator|)
    _SP_DEFINE_BINARY_FUNCTION(bitwise_left_shift, operator<<)
    _SP_DEFINE_BINARY_FUNCTION(bitwise_right_shifit, operator>>)

    _SP_DEFINE_UNARY_FUNCTION(unary_plus, operator+)
    _SP_DEFINE_UNARY_FUNCTION(unary_minus, operator-)

    _SP_DEFINE_UNARY_FUNCTION(logical_not, operator!)
    _SP_DEFINE_BINARY_FUNCTION(logical_and, operator&&)
    _SP_DEFINE_BINARY_FUNCTION(logical_or, operator||)

    _SP_DEFINE_UNARY_FUNCTION(cos, cos)
    _SP_DEFINE_UNARY_FUNCTION(acos, acos)
    _SP_DEFINE_UNARY_FUNCTION(cosh, cosh)
    _SP_DEFINE_UNARY_FUNCTION(sin, sin)
    _SP_DEFINE_UNARY_FUNCTION(asin, asin)
    _SP_DEFINE_UNARY_FUNCTION(sinh, sinh)
    _SP_DEFINE_UNARY_FUNCTION(tan, tan)
    _SP_DEFINE_UNARY_FUNCTION(tanh, tanh)
    _SP_DEFINE_UNARY_FUNCTION(atan, atan)
    _SP_DEFINE_UNARY_FUNCTION(exp, exp)
    _SP_DEFINE_UNARY_FUNCTION(log, log)
    _SP_DEFINE_UNARY_FUNCTION(log10, log10)
    _SP_DEFINE_UNARY_FUNCTION(sqrt, sqrt)
    _SP_DEFINE_BINARY_FUNCTION(atan2, atan2)
    _SP_DEFINE_BINARY_FUNCTION(pow, pow)

#undef _SP_DEFINE_BINARY_FUNCTION
#undef _SP_DEFINE_UNARY_FUNCTION

#define _SP_DEFINE_BINARY_REDUCTION_FUN_(_TAG_, _REDUCTION_, _FUN_)                                       \
    template <typename TL, typename TR>                                                                   \
    __host__ __device__ auto _FUN_(TL const &lhs, TR const &rhs)                                          \
        ->std::enable_if_t<(traits::is_expression<TL>::value || traits::is_expression<TR>::value), bool>  \
    {                                                                                                     \
        return calculus::reduce_expression<tags::_REDUCTION_>(Expression<tags::_TAG_, TL, TR>(lhs, rhs)); \
    };

    _SP_DEFINE_BINARY_REDUCTION_FUN_(not_equal_to, logical_or, operator!=)
    _SP_DEFINE_BINARY_REDUCTION_FUN_(equal_to, logical_and, operator==)
    _SP_DEFINE_BINARY_REDUCTION_FUN_(less, logical_and, operator<)
    _SP_DEFINE_BINARY_REDUCTION_FUN_(greater, logical_and, operator>)
    _SP_DEFINE_BINARY_REDUCTION_FUN_(less_equal, logical_and, operator<=)
    _SP_DEFINE_BINARY_REDUCTION_FUN_(greater_equal, logical_and, operator>=)

#undef _SP_DEFINE_BINARY_REDUCTION_FUN_

    namespace tags
    {
        struct dot
        {
        };
        struct cross
        {
        };
        struct fma
        {
        };
    } // namespace tags

    namespace calculus
    {

        template <typename TL, typename TR>
        struct RecursiveGetHelper<Expression<tags::cross, TL, TR>>
        {
            static constexpr auto nl = traits::number_of_elements<TL, 1>::value;
            static constexpr auto nr = traits::number_of_elements<TR, 1>::value;
            template <std::size_t I, typename E>
            static decltype(auto) eval(E &&expr)
            {
                return get_r<(((I / nl) + 1) % 3) * nl + (I % nl)>(std::get<0>(expr.m_args_)) *
                           get_r<(((I / nr) + 2) % 3) * nr + (I % nr)>(std::get<1>(expr.m_args_)) -
                       get_r<(((I / nl) + 2) % 3) * nl + (I % nl)>(std::get<0>(expr.m_args_)) *
                           get_r<(((I / nr) + 1) % 3) * nr + (I % nr)>(std::get<1>(expr.m_args_));
            }
        };
    } // namespace calculus
    template <typename TL, typename TR>
    decltype(auto) inner_product(TL const &lhs, TR const &rhs)
    {
        return calculus::reduce_expression<tags::addition>(lhs * rhs);
    }
    template <typename TL, typename TR>
    decltype(auto) dot(TL const &lhs, TR const &rhs)
    {
        return calculus::reduce_expression<tags::addition>(lhs * rhs);
    }

    template <typename TL, typename TR>
    auto cross(TL const &lhs, TR const &rhs)
    {
        return Expression<tags::cross, TL, TR>(lhs, rhs);
    };

    template <typename T0, typename T1, typename T2>
    auto fma(T0 const &a0, T1 const &a1, T2 const &a2, std::enable_if_t<(traits::rank<T0>::value > 0)> *_p = nullptr)
    {
        return Expression<tags::fma, T0, T1, T2>(a0, a1, a2);
    };
    template <typename T0, typename T1, typename T2>
    auto fma(T0 const &a0, T1 const &a1, T2 const &a2,
             std::enable_if_t<(std::is_floating_point<T0>::value)> *_p = nullptr)
    {
        return std::fma(a0, a1, a2);
    };
    template <typename T0, typename T1, typename T2>
    auto fma(T0 const &a0, T1 const &a1, T2 const &a2,
             std::enable_if_t<(!std::is_floating_point<T0>::value) && (traits::rank<T0>::value == 0)> *_p = nullptr)
    {
        return a0 * a1 + a2;
    };
    namespace calculus
    {
        template <typename T0, typename T1, typename T2>
        struct RecursiveGetHelper<Expression<tags::fma, T0, T1, T2>>
        {
            template <std::size_t I, typename E>
            static decltype(auto) eval(E &&expr)
            {
                return fma(get_r<I>(std::get<0>(expr.m_args_)), get_r<I>(std::get<1>(expr.m_args_)),
                           get_r<I>(std::get<2>(expr.m_args_)));
            }
        };
    } // namespace calculus

    // template <typename T0>
    // auto sum(T0 const &a0, std::enable_if_t<(traits::rank<T0>::value == 0)> *_p = nullptr) {
    //    return a0;
    //};
    template <typename T0>
    auto sum_t(T0 const &a0)
    {
        return calculus::reduce_expression<tags::addition>(a0);
    };
    template <typename T0>
    auto product_t(T0 const &a0)
    {
        return calculus::reduce_expression<tags::multiplication>(a0);
    };

    template <typename T>
    T determinant(T const &m,
                  std::enable_if_t<(traits::rank<T>::value == 1 && traits::extent<T>::value == 3)> *_p = nullptr)
    {
        return m[0] * m[1] * m[2];
    }

    template <typename T>
    T determinant(T const &m,
                  std::enable_if_t<(traits::rank<T>::value == 1 && traits::extent<T>::value == 4)> *_p = nullptr)
    {
        return m[0] * m[1] * m[2] * m[3];
    }

    template <typename T>
    T determinant(T const &m,
                  std::enable_if_t<(traits::rank<T>::value == 2 && traits::extent<T, 0>::value == 3 &&
                                    traits::extent<T, 1>::value == 3)> *_p = nullptr)
    {
        return m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] * m[1][2] * m[2][0] -
               m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1] * m[0][2] - m[1][2] * m[2][1] * m[0][0];
    }

    template <typename T>
    T determinant(T const &m,
                  std::enable_if_t<(traits::rank<T>::value == 2 && traits::extent<T, 0>::value == 4 &&
                                    traits::extent<T, 1>::value == 4)> *_p = nullptr)
    {
        return m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0] -
               m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3] * m[2][2] * m[3][0] +
               m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1] * m[1][2] * m[2][3] * m[3][0] -
               m[0][3] * m[1][2] * m[2][0] * m[3][1] + m[0][2] * m[1][3] * m[2][0] * m[3][1] +
               m[0][3] * m[1][0] * m[2][2] * m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] -
               m[0][2] * m[1][0] * m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1] +
               m[0][3] * m[1][1] * m[2][0] * m[3][2] - m[0][1] * m[1][3] * m[2][0] * m[3][2] -
               m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1] * m[3][2] +
               m[0][1] * m[1][0] * m[2][3] * m[3][2] - m[0][0] * m[1][1] * m[2][3] * m[3][2] -
               m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1] * m[1][2] * m[2][0] * m[3][3] +
               m[0][2] * m[1][0] * m[2][1] * m[3][3] - m[0][0] * m[1][2] * m[2][1] * m[3][3] -
               m[0][1] * m[1][0] * m[2][2] * m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3];
    }
    // template <typename TL, int... NL, typename TR, int... NR>
    // auto abs(nTuple<TL, NL...> const& l, nTuple<TR, NR...> const& r) {
    //    return std::sqrt(inner_product(l, r));
    //}
    // template <typename T, int... N>
    // T abs(nTuple<T, N...> const& m) {
    //    return std::sqrt(inner_product(m, m));
    //}
    //
    // template <typename T, int... N>
    // auto mod(nTuple<T, N...> const& l) {
    //    return std::sqrt(std::abs(inner_product(l, l)));
    //}
    template <typename T>
    auto power2(T const &v)
    {
        return v * v;
    }
    template <typename T>
    auto normal(T const &l, std::enable_if_t<(traits::rank<T>::value > 0)> *_p = nullptr)
    {
        return (l / (std::sqrt(inner_product(l, l))));
    }
    // template <typename T, typename A>
    // auto rotate(T const& v, T const& u, A const& angle,
    //            std::enable_if_t<(traits::rank<T>::value = 1 && traits::rank<A>::value = 0)>* _p = nullptr) {
    //    Real cosA = std::cos(angle);
    //    Real sinA = std::sin(angle);
    //    return (cosA * v + sinA * cross(u, v) + (1 - cosA) * dot(u, v) * u) / dot(u, u);
    //}

} // namespace sp

#endif // SIMPLA_EXPRESSIONTEMPLATE_H
