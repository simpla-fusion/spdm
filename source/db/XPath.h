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

class Path
{
public:
    enum segment_tags
    {
        Null,
        Key,
        Index,
        Slice
    };

    typedef std::variant<
        std::nullptr_t,
        std::string /* key or query*/,
        int /* index */,
        std ::tuple<int, int, int> /* slice */
        >
        PathSegment;

    typedef Path this_type;

    Path() {}

    Path(const Path& prefix);

    Path(Path&& prefix);

    explicit Path(const std::string& path);

    explicit Path(const char* path) : Path(std::string(path)) {}

    explicit Path(int idx) { m_path_.emplace_back(idx); }

    explicit Path(int start, int stop, int step = 1) { m_path_.emplace_back(std::make_tuple(start, stop, step)); }

    explicit Path(const std::vector<PathSegment>::const_iterator& b, const std::vector<PathSegment>::const_iterator& e) : m_path_(b, e) {}

    ~Path() = default;

    Path& operator=(Path const& other)
    {
        Path(other).swap(*this);
        return *this;
    }

    void swap(Path& other) { std::swap(m_path_, other.m_path_); }

    template <typename Segment>
    this_type operator[](const Segment& key) const { return this->join(key); }

    this_type operator/(const std::string& key) const { return this->join(key); }

    bool empty() const { return m_path_.size() == 0; }

    size_t size() const { return m_path_.size(); }

    auto begin() const { return m_path_.begin(); }

    auto end() const { return m_path_.end(); }

    Path prefix() const { return Path(m_path_.begin(), m_path_.end()); }

    const PathSegment& last() const { return m_path_.back(); }

    std::string str() const;

    std::string filename() const;

    std::string extension() const;

    this_type& append() { return *this; }

    template <typename FirstSegment, typename... Segments>
    this_type& append(FirstSegment&& first_seg, Segments&&... segs)
    {
        m_path_.emplace_back(std::forward<FirstSegment>(first_seg));
        return append(std::forward<Segments>(segs)...);
    }

    template <typename... Segments>
    this_type join(Segments&&... segments) const
    {
        this_type res(*this);
        res.append(std::forward<Segments>(segments)...);
        return std::move(res);
    }

    this_type join(const Path& other) const;

private:
    std::vector<PathSegment> m_path_;
};

/**
 *  - URL : @ref  https://www.ietf.org/rfc/rfc3986.txt
 *  - JSON pointer  
 *  - XPath
 * 
*/
class URI
{
public:
    typedef URI this_type;

    URI(const std::string& uri = "");
    URI(const char*);
    URI(int);
    URI(int, int, int);

    URI(const URI&);
    URI(URI&&);
    ~URI();

    void swap(this_type& other)
    {
        std::swap(m_protocol_, other.m_protocol_);
        std::swap(m_authority_, other.m_authority_);
        std::swap(m_query_, other.m_query_);
        std::swap(m_fragment_, other.m_fragment_);
        m_path_.swap(other.m_path_);
    }

    this_type& operator=(const this_type& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    bool empty() const { return m_path_.size() == 0; }

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

private:
    std::string m_protocol_;
    std::string m_authority_;
    std::string m_query_;
    std::string m_fragment_;
    Path m_path_;
};
namespace literals
{
inline Path operator"" _p(const char* s, std::size_t) { return Path(s); }
} // namespace literals
} // namespace sp::db
#endif //SPDB_XPath_h_