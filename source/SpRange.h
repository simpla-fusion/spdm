#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include <stddef.h>
#include <iterator>
#include <functional>

// template <typename _TIterator, typename _Pred>
// class SpIteratorFilter : public _TIterator
// {
//     typedef SpIteratorFilter<_TIterator, _Pred> this_type;
//     typedef _TIterator base_type;
//     typedef _Pred pred_type;
//     using typename base_type::difference_type;
//     using typename base_type::pointer;
//     using typename base_type::reference;
//     using typename base_type::value_type;

//     SpIteratorFilter(base_type &&self, pred_type &&pred) : base_type(std::forward<base_type>(self)),
//                                                            m_pred_(std::forward<pred_type>(pred)){};
//     ~SpIterator() = default;
//     SpIterator(const this_type &other) : base_type(ohter), m_pred_(other.m_pred_) {}
//     SpIterator(this_type &&other) : m_self_(ohter.m_self_) { other.m_self_ = nullptr; }

//     void next()
//     {
//     }

// private:
//     pred_type m_pred_;
// };

// template <typename... Args, typename _Pred>
// auto filter(SpIterator<Args...> &&it, _Pred &&pred) -> SpIteratorFilter<SpIterator<Args...>, _Pred>
// {
//     return SpIteratorFilter<SpIterator<Args...> >( (std::forward<SpIterator<Args...> >(it), std::forward<_Pred>(pred));
// }

// template <typename _Tp, typename _Pred>
// auto filter(_Tp &&p, _Pred &&pred)
// {
//     return filter(iterator(std::forward<_Tp>(p)), std::forward<_Pred>(pred));
// }

#endif //SP_RANGE_H_