#ifndef SP_Path_h_
#define SP_Path_h_

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
namespace sp
{
/**
 *  - URL : @ref  https://www.ietf.org/rfc/rfc3986.txt
 *  - JSON pointer  
 *  - XPath
 * 
*/
class Path
{
public:
    typedef std::variant<std::string /* key or query*/, int /* index */, std ::tuple<int, int, int> /* slice */> segment;

    enum segment_type
    {
        KEY,
        INDEX,
        SLICE
    };

    typedef Path this_type;
    Path(const std::string&);
    Path(const Path&);
    Path(Path&&);
    ~Path();

    template <typename FirstSegment, typename... Others>
    Path(const Path& other, FirstSegment&& seg, Others&&... others)
        : Path(other) { append(std::forward<FirstSegment>(seg), std::forward<Others>(others)...); }

    template <typename FirstSegment, typename... Others>
    Path(Path&& other, FirstSegment&& seg, Others&&... others)
        : Path(std::forward<Path>(other)) { append(std::forward<FirstSegment>(seg), std::forward<Others>(others)...); }

    void swap(this_type& other)
    {
        std::swap(m_scheme_, other.m_scheme_);
        std::swap(m_authority_, other.m_authority_);
        std::swap(m_path_, other.m_path_);
        std::swap(m_query_, other.m_query_);
        std::swap(m_fragment_, other.m_fragment_);
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

    const std::string& scheme() const { return m_scheme_; }
    void scheme(const std::string& s) { m_scheme_ = s; }

    const std::string& authority() const { return m_authority_; }
    void authority(const std::string& s) { m_authority_ = s; }

    const std::string& query() const { return m_query_; }
    void query(const std::string& s) { m_query_ = s; }

    const std::string& fragment() const { return m_fragment_; }
    void fragment(const std::string& s) { m_fragment_ = s; }

    template <typename Key>
    this_type operator[](const Key& key) const { return Path(*this, key); }

    this_type operator/(const std::string& key) const { return Path(*this, key); }

    size_t size() const { return m_path_.size(); }

    auto begin() const { return m_path_.begin(); }

    auto end() const { return m_path_.end(); }

private:
    std::string m_scheme_;
    std::string m_authority_;
    std::vector<segment_type> m_path_;
    std::string m_query_;
    std::string m_fragment_;
};

inline Path operator"" _p(const char* s, std::size_t) { return Path(s); }

} // namespace sp
#endif //SP_Path_h_