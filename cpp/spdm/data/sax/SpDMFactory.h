//
// Created by salmon on 18-3-15.
//

#ifndef SIMPLA_SPDMWRITER_H
#define SIMPLA_SPDMWRITER_H

#include <deque>
#include <functional>
#include "SpDM.h"
namespace sp {

template <typename TObj, typename SFINAE = void>
struct SpDMFactory;

template <typename TObj>
struct SpDMFactory<TObj, std::enable_if_t<std::is_base_of<SpDMObject<>, TObj>::value>>
    : public SpDMProxyVisitor<SpDMElement<>, typename SpDMElement<>::char_type> {
    typedef SpDMProxyVisitor<SpDMElement<>, typename SpDMElement<>::char_type> base_type;
    typedef SpDMFactory<TObj> this_type;
    typedef SpDMObject<> dom_type;
    typedef TObj supper_class;
    typedef typename dom_type::size_type size_type;
    typedef typename dom_type::char_type char_type;
    typedef typename dom_type::key_type key_type;
    typedef typename dom_type::value_type value_type;
    typedef typename dom_type::number_type number_type;
    typedef typename dom_type::string_type string_type;
    typedef typename dom_type::array_type array_type;
    typedef typename dom_type::object_type object_type;

    typedef typename base_type::interface_type interface_type;

   protected:
    key_type m_tag_;
    value_type m_holder_;

   public:
    explicit SpDMFactory(value_type *buffer) : base_type(buffer) {}
    explicit SpDMFactory(value_type *buffer, key_type tag) : base_type(buffer), m_tag_(std::move(tag)) {}

    template <typename U, typename SFINAE = std::enable_if_t<std::is_base_of<object_type, U>::value>>
    explicit SpDMFactory(U &dom) : m_holder_(&dom, kIsReference), base_type(&m_holder_) {}

    ~SpDMFactory() override {
        switch (m_buffer_->flag().type & kTypeMask) {
            case kObject: {
                auto const &f = utility::SingletonHolder<ObjectFactory>::instance().m_factory_;
                supper_class *res = nullptr;
                auto it = f.find(m_tag_);
                if (it != f.end()) { value_type(New(m_tag_, std::move(*m_buffer_))).swap(*m_buffer_); }
            } break;
            case kArray: {
                //                m_buffer_->Number();
                break;
            }
            case kBool:
            case kNumber:
            case kChar:
            default:
                break;
        }
    };

    interface_type *Array() override { return new this_type(base_type::m_buffer_); };
    interface_type *Object() override { return new this_type(base_type::m_buffer_); };
    interface_type *Add() override { return new this_type(base_type::m_buffer_->Add()); };
    interface_type *Insert(char_type const *str, size_type len) override {
        return new this_type(base_type::m_buffer_->Insert(str, len), key_type(str, len));
    }

    struct ObjectFactory {
        std::map<key_type, std::function<supper_class *(value_type)>> m_factory_;
    };
    static bool HasCreator(key_type const &tag_name) {
        auto const &f = utility::SingletonHolder<ObjectFactory>::instance().m_factory_;
        return f.find(tag_name) != f.end();
    }

    static int RegisterCreator(key_type const &tag_name,
                               std::function<supper_class *(value_type)> const &fun) noexcept {
        return utility::SingletonHolder<ObjectFactory>::instance().m_factory_.emplace(tag_name, fun).second ? 1 : 0;
    };
    template <typename U>
    static int RegisterCreator(key_type const &tag_name,
                               std::enable_if_t<(std::is_constructible<U, value_type>::value)> *_p = nullptr) noexcept {
        return RegisterCreator(tag_name, [](value_type v) { return new U(std::move(v)); });
    };
    template <typename U>
    static int RegisterCreator(
        key_type const &tag_name,
        std::enable_if_t<(std::is_constructible<U>::value) && (!std::is_constructible<U, value_type>::value)> *_p =
            nullptr) noexcept {
        return RegisterCreator(tag_name, [](value_type v) -> supper_class * {
            auto res = new U();
            res->Deserialize(std::move(v));
            return dynamic_cast<supper_class *>(res);
        });
    };
    template <typename U>
    static int RegisterCreator(std::string const &tag_name) noexcept {
        return RegisterCreator<U>(key_type(tag_name));
    }
    template <typename U>
    static int RegisterCreator() noexcept {
        return RegisterCreator<U>(key_type(U::RegisterName()));
    }

   public:
    static supper_class *New(key_type const &tag, value_type dm) {
        auto const &f = utility::SingletonHolder<ObjectFactory>::instance().m_factory_;
        supper_class *res = nullptr;
        auto it = f.find(tag);
        if (it != f.end()) { res = it->second(std::move(dm)); }
        //        else {
        //            std::cout << tag.template as<std::string>() << " is not registered ! [ ";
        //            for (auto const &item : f) { std::cout << item.first.c_str() << "  " << std::endl; }
        //            std::cout << "]" << std::endl;
        //            ERR_OUT_OF_RANGE("");
        //        }
        return res;
    }
    static auto New(value_type const &dm) { return New(dm.at("@type").asString(), dm); }

    template <typename K, typename... Args>
    static auto New(K const &tag, Args &&... args) {
        return New(key_type(tag), value_type(std::forward<Args>(args)...));
    }
};

template <typename TBase, typename TObj>
struct SpDMRegisteredInFactory {
    static int s_is_registered_;
};
template <typename TBase, typename TObj>
int SpDMRegisteredInFactory<TBase, TObj>::s_is_registered_ =
    ::sp::SpDMFactory<TBase>::template RegisterCreator<TObj>(TObj::RegisterName());

#define SP_OBJECT_REGISTER(...)

#define SP_REGISTERED_IN_FACTORY(_BASE_, _CLASS_) sp::SpDMRegisteredInFactory<_BASE_, _CLASS_>

#define SP_REGISTER_OBJECT_2(_BASE_, _CLASS_)                                  \
    SP_OBJECT_HEAD(_BASE_, _CLASS_)                                            \
    template <typename... Args>                                                \
    static auto New(Args &&... args) {                                         \
        return new _CLASS_(std::forward<Args>(args)...);                       \
    }                                                                          \
    this_type *Copy() const override { return new this_type(*this); };         \
    std::string GetRegisterName() const override { return __STRING(_CLASS_); } \
    static std::string RegisterName() {                                        \
        sp::SpDMRegisteredInFactory<_BASE_, _CLASS_>::s_is_registered_;    \
        return __STRING(_CLASS_);                                              \
    }

#define SP_REGISTER_OBJECT_3(_BASE_, _CLASS_, _REG_NAME_)                         \
    SP_OBJECT_HEAD(_BASE_, _CLASS_)                                               \
    template <typename... Args>                                                   \
    static auto New(Args &&... args) {                                            \
        return new _CLASS_(std::forward<Args>(args)...);                          \
    }                                                                             \
    this_type *Copy() const override { return new this_type(*this); };            \
    std::string GetRegisterName() const override { return __STRING(_REG_NAME_); } \
    static std::string RegisterName() {                                           \
        sp::SpDMRegisteredInFactory<_BASE_, _CLASS_>::s_is_registered_;       \
        return __STRING(_REG_NAME_);                                              \
    }

#define SP_REGISTERED_OBJECT_HEAD(...) VFUNC(SP_REGISTER_OBJECT_, __VA_ARGS__)

#define SP_SUPPER_OBJECT_HEAD(_CLASS_)                                                                  \
                                                                                                        \
   private:                                                                                             \
    typedef spObject base_type;                                                                         \
    typedef _CLASS_ this_type;                                                                          \
                                                                                                        \
   public:                                                                                              \
    std::string GetRegisterName() const override { return __STRING(_CLASS_); }                          \
                                                                                                        \
    using typename base_type::visitor_type;                                                             \
    using typename base_type::data_entry_type;                                                          \
    _CLASS_ *Copy() const override { return nullptr; }                                                  \
    template <typename _U>                                                                              \
    static auto New(_U *p, std::enable_if_t<std::is_base_of<_CLASS_, _U>::value, _U *> *_p = nullptr) { \
        return p;                                                                                       \
    }                                                                                                   \
    template <typename _U>                                                                              \
    static auto New(_U const *p, std::enable_if_t<std::is_base_of<_CLASS_, _U>::value> *_p = nullptr) { \
        return p->Copy();                                                                               \
    }                                                                                                   \
    template <typename... Args>                                                                         \
    static _CLASS_ *New(Args &&... args) {                                                              \
        return sp::SpDMFactory<_CLASS_>::New(std::forward<Args>(args)...);                          \
    }

}  // namespace sp
#endif  // SIMPLA_SPDMWRITER_H
