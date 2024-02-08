//
// Created by salmon on 18-2-19.
//

#ifndef SIMPLA_SPDMXDMF_H
#define SIMPLA_SPDMXDMF_H

#include <boost/optional.hpp>
#include <deque>
#include <memory>
#include "SpDM.h"
namespace sp {
using boost::optional;

namespace data {
template <typename OS>
struct SpDMXDMF : public SpDMSAXInterface {
    OS &m_os_;

    int m_tab_width_ = 4;
    int m_indent_ = 0;
    bool m_is_attribute_ = false;
    struct node_s {
        std::string tag;
        bool is_array = false;
    };
    std::string m_tag_;
    std::deque<node_s> m_stack_;

    explicit SpDMXDMF(OS &o, int default_tab_width = 4) : m_os_(o), m_tag_("root"), m_tab_width_(default_tab_width) {}
    ~SpDMXDMF() override {
        while (!m_stack_.empty()) { Pop(); }
    };
    bool Null() override {
        m_os_ << "<" << m_tag_ << "/>" << std::endl;
        return true;
    }
    bool Bool(bool b) override { return Put(b); }
    bool Int(int i) override { return Put(i); }
    bool Uint(unsigned u) override { return Put(u); }
    bool Int64(int64_t i) override { return Put(i); }
    bool Uint64(uint64_t u) override { return Put(u); }
    bool Double(double d) override { return Put(d); }
    bool String(const char *str, size_t length = 0, bool copy = true) override { return Put(str, length); }

    bool TensorBool(bool const *b, unsigned int rank, size_t const *dims) override { return Put(b, rank, dims); }
    bool TensorInt(int const *i, unsigned int rank, size_t const *dims) override { return Put(i, rank, dims); }
    bool TensorUint(unsigned const *u, unsigned int rank, size_t const *dims) override { return Put(u, rank, dims); }
    bool TensorInt64(int64_t const *i, unsigned int rank, size_t const *dims) override { return Put(i, rank, dims); }
    bool TensorUint64(uint64_t const *u, unsigned int rank, size_t const *dims) override { return Put(u, rank, dims); }
    bool TensorDouble(double const *d, unsigned int rank, size_t const *dims) override { return Put(d, rank, dims); }
//    bool Object(SpDMObject const *) override { return true; };

    bool StartObject() override {
        Push(m_tag_, false);
        m_is_attribute_ = true;
        return true;
    }
    bool Key(const char *tag, size_t length = 0, bool copy = true) override {
        if (tag[0] == '@') {
            m_tag_ = tag + 1;
            m_is_attribute_ = true;
        } else {
            if (m_is_attribute_) { m_os_ << ">"; }
            m_tag_ = tag;
            m_is_attribute_ = false;
        }
        return true;
    }
    bool EndObject(size_t memberCount) override {
        Pop();
        return true;
    }
    bool StartArray() override { return Push(m_tag_, true); }
    bool EndArray(size_t elementCount) override { return Pop(); }

   private:
    bool Push(std::string const &tag, bool is_array) {
        if (!is_array) {
            m_os_ << std::endl << std::setw(m_indent_) << "<" << m_tag_;
            m_indent_ += m_tab_width_;
        }
        m_stack_.emplace_front(node_s{tag, is_array});
        return true;
    }
    bool Pop() {
        if (!m_stack_.begin()->is_array) {
            m_indent_ -= m_tab_width_;
            m_os_ << std::endl << std::setw(m_indent_) << "</" << m_stack_.begin()->tag << ">";
            m_is_attribute_ = false;
        }
        m_stack_.pop_front();
        if (m_stack_.empty()) {
            m_tag_.clear();
        } else {
            m_tag_ = m_stack_.begin()->tag;
        }
        return true;
    }

    template <typename V>
    bool Put(V const &v) {
        if (m_is_attribute_) {
            m_os_ << " " << m_tag_ << "=\"" << std::boolalpha << v << "\"";
        } else {
            m_os_ << std::endl
                  << std::setw(m_indent_) << "<" << m_tag_ << ">" << std::boolalpha << v << "</" << m_tag_ << ">";
        }
        return true;
    }
    bool Put(const char *str, unsigned int length = 0) { return Put(std::string(str)); }
    template <typename V>
    bool Put(V const *v, unsigned int rank, size_t const *dims) {
        if (m_is_attribute_) {
            m_os_ << ">";
            m_is_attribute_ = false;
        }
        m_os_ << std::endl << std::setw(m_indent_) << "<" << m_tag_ << R"( Format="XML" Dimensions=")";
        m_os_ << dims[0];
        for (int i = 1; i < rank; ++i) { m_os_ << " " << dims[i]; }
        m_os_ << R"(">)";
        utility::FancyPrintP(m_os_, v, rank, dims, m_indent_ + m_tab_width_, m_tab_width_, ' ', ' ', ' ');
        if (rank > 1) { m_os_ << std::endl << std::setw(m_indent_); }
        m_os_ << "</" << m_tag_ << ">";
        return true;
    }
};

template <typename DM, typename SFINAE = std::enable_if_t<traits::is_spdm<DM>::value>>
DM ReadXDMF(std::string const &text) {
    DM db;
    //    detail::ReadXMLHandler<DM> handler(db);
    //    rapidjson::Reader reader;
    //    rapidjson::InsituStringStream ss(const_cast<char *>(json.c_str()));
    //    reader.Parse(ss, handler);
    return std::move(db);
}

template <typename DM, typename OS, typename SFINAE = std::enable_if_t<traits::is_spdm<DM>::value>>
OS &WriteXDMF(DM const &db, OS &os) {
    SpDMXDMF<OS> xdmf(os);
    db.Accept(xdmf);
    return os;
}

}  // namespace m_data_
}  // namespace sp
#endif  // SIMPLA_SPDMXDMF_H
