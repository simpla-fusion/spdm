//
// Created by salmon on 18-2-13.
//

#ifndef SIMPLA_SPDMIOSTREAM_H
#define SIMPLA_SPDMIOSTREAM_H

#include "SpDM.h"
#include "Utility.h"

//#ifndef NO_JSON
//#include "SpDMJSON.h"
//#endif

namespace sp {
template <typename OS, typename _C>
struct SpDMProxyVisitor<OS, _C, std::enable_if_t<std::is_base_of<std::basic_ostream<_C>, OS>::value>>
    : public SpDMVisitorInterface<_C> {
    typedef SpDMProxyVisitor this_type;
    typedef SpDMVisitorInterface<_C> interface_type;
    typedef _C char_type;
    typedef size_t size_type;
    OS &os;
    int m_indent_ = 0;
    int m_tab_width_ = 4;
    int m_type_ = kNull;
    int m_index_num_ = 0;

    explicit SpDMProxyVisitor(OS &o, int type = kNull, int indent = 0, int default_tab = 2)
        : os(o), m_type_(type), m_index_num_(0), m_indent_(indent), m_tab_width_(default_tab) {
        switch (m_type_) {
            case kArray:
                os << "[";
                break;
            case kObject:
                os << "{";
                break;
            default:
                break;
        }
    }
    SpDMProxyVisitor(const SpDMProxyVisitor &other)
        : os(const_cast<OS &>(other.os)),
          m_indent_(other.m_indent_),
          m_tab_width_(other.m_tab_width_),
          m_type_(other.m_type_),
          m_index_num_(other.m_index_num_) {}
    SpDMProxyVisitor(SpDMProxyVisitor &&other) noexcept
        : os(const_cast<OS &>(other.os)),
          m_indent_(other.m_indent_),
          m_tab_width_(other.m_tab_width_),
          m_type_(other.m_type_),
          m_index_num_(other.m_index_num_) {}
    ~SpDMProxyVisitor() override {
        switch (m_type_) {
            case kArray:
                os << "]";
                break;
            case kObject:
                os << std::endl << std::setw(m_indent_) << "}";
                break;
            default:
                break;
        }
    };

    interface_type *Copy() const override { return new this_type(*this); }

    interface_type *Array() override { return new this_type(os, kArray, m_indent_, m_tab_width_); };
    interface_type *Object() override { return new this_type(os, kObject, m_indent_, m_tab_width_); };
    interface_type *Add() override {
        if (m_index_num_ > 0) { os << ", " << std::setw(m_indent_) << " "; }
        ++m_index_num_;
        return new this_type(os, kNull, m_indent_ + m_tab_width_, m_tab_width_);
    }
    interface_type *Insert(char_type const *str, size_type len = 0) override {
        if (m_index_num_ > 0) { os << ", "; }
        os << std::endl
           << std::setw(m_indent_) << " "
           << "\"" << str << "\": ";
        ++m_index_num_;
        return new this_type(os, kNull, m_indent_ + m_tab_width_, m_tab_width_);
    };
    Status Null() override {
        os << "null";
        return Status::OK();
    }
    Status Set() override { return Status::OK(); }
    Status Set(bool b) override {
        os << std::boolalpha << b;
        return Status::OK();
    }
    Status Set(int v) override {
        os << v;
        return Status::OK();
    }
    Status Set(unsigned v) override {
        os << v;
        return Status::OK();
    }
    Status Set(int64_t v) override {
        os << v;
        return Status::OK();
    }
    Status Set(uint64_t v) override {
        os << v;
        return Status::OK();
    }
    Status Set(double v) override {
        os << v;
        return Status::OK();
    }
    Status Set(char_type const *str, size_type length = 0) override {
        os << "\"" << str << "\"";
        return Status::OK();
    }
    Status Set(bool const *b, unsigned int rank, size_t const *dims) override { return Put(b, rank, dims); };
    Status Set(int const *i, unsigned int rank, size_t const *dims) override { return Put(i, rank, dims); };
    Status Set(unsigned const *u, unsigned int rank, size_t const *dims) override { return Put(u, rank, dims); };
    Status Set(int64_t const *i, unsigned int rank, size_t const *dims) override { return Put(i, rank, dims); };
    Status Set(uint64_t const *u, unsigned int rank, size_t const *dims) override { return Put(u, rank, dims); };
    Status Set(double const *d, unsigned int rank, size_t const *dims) override { return Put(d, rank, dims); };

   private:
    template <typename V>
    Status Put(V const *v, unsigned int rank, size_t const *dims) {
        utility::FancyPrintP(os, v, rank, dims, m_indent_, m_tab_width_, '[', ',', ']');
        return Status::OK();
    }
};

template <typename _C>
std::ostream &operator<<(std::ostream &os, SpDMElement<_C> const &v) {
    v.Serialize(os);
    return os;
}
template <typename... T>
std::ostream &operator<<(std::ostream &os, SpDMObject<T...> const &v) {
    v.Serialize(os);
    return os;
}
template <typename C>
std::ostream &operator<<(std::ostream &os, SpDMString<C> const &s) {
    os << s.c_str();
    return os;
}
template <typename _C>
std::istream &operator>>(std::istream &is, SpDMElement<_C> &v) {
#ifndef NO_JSON
//    std::string s(std::istreambuf_iterator<char>(is), {});
//    v.Set(ReadJSON<SpDMElement<_C>>(s));
#else
#endif
    throw(std::runtime_error("unimlemented!"));
    return is;
}
}  // namespace sp {
#endif  // SIMPLA_SPDMIOSTREAM_H
