//
// Created by salmon on 18-3-20.
//

#ifndef SIMPLA_SPDMSAXINTERFACE_H
#define SIMPLA_SPDMSAXINTERFACE_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include "SpDM.h"
namespace sp {
namespace traits {
template <typename U, typename SFINAE = void>
struct is_sax_interface : public std::false_type {};
}

template <typename Handler, typename _C>
struct SpDMProxyVisitor<Handler, _C, std::enable_if_t<traits::is_sax_interface<Handler>::value>>
    : public SpDMVisitorInterface<_C> {
    typedef SpDMProxyVisitor<Handler, _C> this_type;
    typedef SpDMVisitorInterface<_C> interface_type;
    typedef _C char_type;
    typedef std::size_t size_type;
    Handler &m_handler_;
    int m_type_ = kNull;
    int m_count_ = 0;
    SpDMProxyVisitor(Handler &handler, int type = kNull) : m_handler_(handler), m_type_(type) {
        switch (m_type_) {
            case kObject:
                m_handler_.StartObject();
                break;
            case kArray:
                m_handler_.StartArray();
                break;
        }
    }
    ~SpDMProxyVisitor() override {
        switch (m_type_) {
            case kObject:
                m_handler_.EndObject(m_count_);
                break;
            case kArray:
                m_handler_.EndArray(m_count_);
                break;
        }
    }
    SpDMProxyVisitor(SpDMProxyVisitor const &other)
        : m_handler_(other.m_handler_), m_type_(other.m_type_), m_count_(other.m_count_) {}

    SpDMProxyVisitor(SpDMProxyVisitor &&other)
        : m_handler_(other.m_handler_), m_type_(other.m_type_), m_count_(other.m_count_) {}

    interface_type *Copy() const override { return new this_type(*this); }
    int Set() override { return m_handler_.Null(); };
    int Set(bool b) override { return m_handler_.Bool(b); };
    int Set(int i) override { return m_handler_.Int(i); };
    int Set(unsigned u) override { return m_handler_.Uint(u); };
    int Set(int64_t i) override { return m_handler_.Int64(i); };
    int Set(uint64_t u) override { return m_handler_.Uint64(u); };
    int Set(double d) override { return m_handler_.Double(d); };
    int Set(char_type const *str, size_type length) override { return m_handler_.String(str, length, true); };
    int Set(bool const *b, unsigned int rank, size_type const *dims) override { return PutTensor(b, rank, dims); };
    int Set(int const *i, unsigned int rank, size_type const *dims) override { return PutTensor(i, rank, dims); };
    int Set(unsigned const *u, unsigned int rank, size_type const *dims) override { return PutTensor(u, rank, dims); };
    int Set(int64_t const *i, unsigned int rank, size_type const *dims) override { return PutTensor(i, rank, dims); };
    int Set(uint64_t const *u, unsigned int rank, size_type const *dims) override { return PutTensor(u, rank, dims); };
    int Set(double const *d, unsigned int rank, size_type const *dims) override { return PutTensor(d, rank, dims); };

    template <typename V>
    int PutTensor(V const *v, unsigned int rank, size_t const *dims) {
        if (v == nullptr) {
            return false;
        } else if (rank == 0) {
            return Set(*v);
        }
        int count = 0;
        size_t idx[rank];
        for (int i = 0; i < rank; ++i) { idx[i] = 0; }
        int n = 0;
        size_t pos = 0;
        count += m_handler_.StartArray();
        while (true) {
            for (int i = 0; i < rank - 1; ++i) {
                if (idx[i + 1] == 0) { count += m_handler_.StartArray(); }
            }
            for (auto s = pos, se = pos + dims[rank - 1]; s < se; ++s) { count += Set(*(v + s)); }
            pos += dims[rank - 1];
            idx[rank - 1] = dims[rank - 1];
            for (int s = rank - 1; s > 0; --s) {
                if (idx[s] >= dims[s]) {
                    count += m_handler_.EndArray(dims[s]);
                    ++idx[s - 1];
                    idx[s] = 0;
                }
            }
            if (idx[0] >= dims[0]) { break; }
        }
        count += m_handler_.EndArray(dims[0]);

        return count;
    };

    int Null() override { return m_handler_.Null(); };
    interface_type *Array() override { return new this_type(m_handler_, kArray); };
    interface_type *Object() override { return new this_type(m_handler_, kObject); };
    interface_type *Add() override {
        ++m_count_;
        return new this_type(m_handler_, kNull);
    };
    interface_type *Insert(char_type const *str, size_type len) override {
        ++m_count_;
        m_handler_.Key(str, len, true);
        return new this_type(m_handler_, kNull);
    }
};

template <typename TObj, typename SIFNAE = void>
struct SpDMSAXWrapper;
template <typename TObj>
struct SpDMSAXWrapper<TObj, std::enable_if_t<traits::is_spdm<TObj>::value>> {
    typedef TObj object_type;
    typedef typename object_type::value_type *reference_type;
    typedef typename object_type::size_type size_type;
    typedef typename object_type::char_type char_type;

   protected:
    std::deque<reference_type> m_stack_;
    std::string m_tag_;

   public:
    template <typename... Args>
    explicit SpDMSAXWrapper(Args &&... args) {
        m_stack_.emplace_front(std::forward<Args>(args)...);
    }
    template <typename... Args>
    explicit SpDMSAXWrapper(TObj &obj) {
        m_stack_.emplace_front(&obj);
    }
    ~SpDMSAXWrapper() = default;
    int PutValue() {
        m_stack_.pop_front();
        return 1;
    }
    template <typename... Args>
    int PutValue(Args &&... args) {
        int count = 0;
        if (m_stack_.front()->isArray()) {
            count = m_stack_.front()->Add()->Set(std::forward<Args>(args)...);
        } else {
            count = m_stack_.front()->Set(std::forward<Args>(args)...);
            m_stack_.pop_front();
        }
        return count;
    }

    bool Null() { return PutValue() > 0; }
    bool Bool(bool b) { return PutValue(b) > 0; }
    bool Int(int i) { return PutValue(i) > 0; }
    bool Uint(unsigned u) { return PutValue(u) > 0; }
    bool Int64(int64_t i) { return PutValue(i) > 0; }
    bool Uint64(uint64_t u) { return PutValue(u) > 0; }
    bool Double(double d) { return PutValue(d) > 0; }
    bool String(const char *str, size_type length, bool copy) { return PutValue(str, length, !copy) > 0; }

    bool StartObject() {
        if (m_stack_.front()->isArray()) {
            m_stack_.emplace_front(m_stack_.front()->Add()->Object());
        } else {
            m_stack_.front()->Object();
        }
        return true;
    }
    bool Key(const char_type *str, size_type length, bool copy) {
        m_tag_ = str;
        m_stack_.emplace_front(m_stack_.front()->Insert(str));
        return true;
    }
    bool EndObject(size_type memberCount) {
        m_stack_.front()->Object();
        m_stack_.pop_front();
        return true;
    }
    bool StartArray() {
        if (m_stack_.front()->isArray()) {
            m_stack_.emplace_front(m_stack_.front()->Add()->Array());
        } else {
            m_stack_.front()->Array();
        }
        return true;
    }
    bool EndArray(size_type elementCount) {
        auto it = m_stack_.begin();
        ++it;
        if (it != m_stack_.end() && !(*it)->isArray()) { m_stack_.front()->Number(); }
        m_stack_.pop_front();
        return true;
    }

    bool RawNumber(const char *str, size_t len, bool copy) { return String(str, len, copy); }
};

}  // namespace sp{

#endif  // SIMPLA_SPDMSAXINTERFACE_H
