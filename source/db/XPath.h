#ifndef SPDB_XPath_h_
#define SPDB_XPath_h_

#include <algorithm>
#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdlib.h>
#include <string>
#include <variant>
#include <vector>
namespace sp::db
{
/**
 *  - URL : @ref  https://www.ietf.org/rfc/rfc3986.txt
 *  - JSON pointer  
 *  - XXPath
 * 
*/
class XPath
{
public:
    typedef std::variant<std::string /* key or query*/, int /* index */, std ::tuple<int, int, int> /* slice */> element;

    enum type_tags
    {
        Key,
        Index,
        Slice
    };

    typedef XPath this_type;
    XPath(const std::string&);
    XPath(const XPath&);
    XPath(XPath&&);
    ~XPath();

    template <typename FirstSegment, typename... Others>
    XPath(const XPath& other, FirstSegment&& seg, Others&&... others)
        : XPath(other) { append(std::forward<FirstSegment>(seg), std::forward<Others>(others)...); }

    template <typename FirstSegment, typename... Others>
    XPath(XPath&& other, FirstSegment&& seg, Others&&... others)
        : XPath(std::forward<XPath>(other)) { append(std::forward<FirstSegment>(seg), std::forward<Others>(others)...); }

    void swap(this_type& other)
    {
        std::swap(m_protocol_, other.m_protocol_);
        std::swap(m_authority_, other.m_authority_);
        std::swap(m_path_, other.m_path_);
        std::swap(m_query_, other.m_query_);
        std::swap(m_fragment_, other.m_fragment_);
        std::swap(m_uri_, other.m_uri_);
    }

    this_type& operator=(const this_type& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void append(const std::string& path);
    void append(int idx);
    void append(int b, int e, int seq = 1);

    std::string str() const;

    operator std::string() const { return str(); }

    const std::string& protocol() const { return m_protocol_; }
    void protocol(const std::string& s) { m_protocol_ = s; }

    const std::string& authority() const { return m_authority_; }
    void authority(const std::string& s) { m_authority_ = s; }

    const std::string& query() const { return m_query_; }
    void query(const std::string& s) { m_query_ = s; }

    const std::string& fragment() const { return m_fragment_; }
    void fragment(const std::string& s) { m_fragment_ = s; }

    std::string filename() const;
    std::string extension() const;

    template <typename Key>
    this_type operator[](const Key& key) const { return XPath(*this, key); }

    this_type operator/(const std::string& key) const { return XPath(*this, key); }

    size_t size() const { return m_path_.size(); }

    auto begin() const { return m_path_.begin(); }

    auto end() const { return m_path_.end(); }

private:
    std::string m_protocol_;
    std::string m_authority_;
    std::vector<element> m_path_;
    std::string m_query_;
    std::string m_fragment_;
    std::string m_uri_;
};
namespace literals
{
inline XPath operator"" _p(const char* s, std::size_t) { return XPath(s); }
} // namespace literals
} // namespace sp::db
#endif //SPDB_XPath_h_