//
// Created by salmon on 18-2-9.
//

#ifndef SIMPLA_SPDMXML_H
#define SIMPLA_SPDMXML_H

#include <deque>
#include "SpDMSAX.h"
namespace simpla {

template <typename OS>
struct SpDMXMLWriter {
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

    explicit SpDMXMLWriter(OS &o, int default_tab_width = 4)
        : m_os_(o), m_tag_("root"), m_tab_width_(default_tab_width) {}
    ~SpDMXMLWriter() {
        while (!m_stack_.empty()) { Pop(); }
    };
    bool Null() {
        m_os_ << "<" << m_tag_ << "/>" << std::endl;
        return true;
    }
    bool Bool(bool b) { return Put(b); }
    bool Int(int i) { return Put(i); }
    bool Uint(unsigned u) { return Put(u); }
    bool Int64(int64_t i) { return Put(i); }
    bool Uint64(uint64_t u) { return Put(u); }
    bool Double(double d) { return Put(d); }
    bool String(const char *str, size_t length = 0, bool copy = true) { return Put(str, length); }

    //    bool TensorBool(bool const *b, unsigned int rank, size_t const *dims)  { return Put(b, rank, dims); }
    //    bool TensorInt(int const *i, unsigned int rank, size_t const *dims)  { return Put(i, rank, dims); }
    //    bool TensorUint(unsigned const *u, unsigned int rank, size_t const *dims)  { return Put(u, rank,
    //    dims); }
    //    bool TensorInt64(int64_t const *i, unsigned int rank, size_t const *dims)  { return Put(i, rank,
    //    dims); }
    //    bool TensorUint64(uint64_t const *u, unsigned int rank, size_t const *dims)  { return Put(u, rank,
    //    dims); }
    //    bool TensorDouble(double const *d, unsigned int rank, size_t const *dims)  { return Put(d, rank,
    //    dims); }
    //    bool Object(SpDMObject const *)  { return true; };

    bool StartObject() {
        Push(m_tag_, false);
        m_is_attribute_ = true;
        return true;
    }
    bool Key(const char *tag, size_t length = 0, bool copy = true) {
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
    bool EndObject(size_t memberCount) {
        Pop();
        return true;
    }
    bool StartArray() { return Push(m_tag_, true); }
    bool EndArray(size_t elementCount) { return Pop(); }

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

template <typename DOM>
DOM ReadXML(char const *text) {
    DOM db;
    //    detail::ReadXMLHandler<DM> handler(db);
    //    rapidjson::Reader reader;
    //    rapidjson::InsituStringStream ss(const_cast<char *>(json.c_str()));
    //    reader.Parse(ss, handler);
    return std::move(db);
}
template <typename IS, typename DOM = SpDataEntry>
auto ReadXML(IS &input_stream) {
    std::string json(std::istreambuf_iterator<char>(input_stream), {});
    return ReadXML<DOM>(json.c_str());
}
template <typename OS, typename DOM>
OS &WriteXML(OS &os, DOM const &db) {
    SpDMXMLWriter<OS> xml_writer(os);
    db.Accept(xml_writer);
    return os;
}

namespace traits {
template <typename OS>
struct is_sax_interface<SpDMXMLWriter<OS>> : public std::true_type {};
}
}  // namespace simpla

#endif  // SIMPLA_SPDMXML_H
