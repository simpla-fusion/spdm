#ifndef SPDB_XPath_h_
#define SPDB_XPath_h_

#include <algorithm>
#include <any>
#include <functional>
#include <iostream>
#include <list>
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
        Segment;

    typedef Path this_type;

    static Path parse(const std::string&);

    Path();

    ~Path() = default;

    Path(const Path& prefix);

    Path(Path&& prefix);

    template <typename... Args>
    explicit Path(Args&&... args) : Path() { append(std::forward<Args>(args)...); }

    Path& operator=(Path const& other)
    {
        Path(other).swap(*this);
        return *this;
    }

    void swap(Path& other) { std::swap(m_path_, other.m_path_); }

    // template <typename Segment>
    // this_type operator[](const Segment& key) const { return this->join(key); }

    template <typename U>
    this_type operator/(const U& key) const { return join(*this, key); }

    bool empty() const { return m_path_->size() == 0; }

    void clear() { m_path_->clear(); }

    size_t size() const { return m_path_->size(); }

    auto begin() const { return m_path_->begin(); }

    auto end() const { return m_path_->end(); }

    Path prefix() const;

    const Segment& last() const { return m_path_->back(); }

    std::string str() const;

    std::string filename() const;

    std::string extension() const;

    this_type& append(const this_type& other);

    // this_type& append(const std::string& uri);

    template <typename U>
    this_type& append(const U& seg)
    {
        m_path_->emplace_back(seg);
        return *this;
    }

    template <typename First, typename Second, typename... Others>
    this_type& append(const First&& first, Second&& second, Others&&... others)
    {
        return append(std::forward<First>(first)).append(std::forward<Second>(second), std::forward<Others>(others)...);
    }

    template <typename... Components>
    Path join(Components&&... comp) &&
    {
        append(std::forward<Components>(comp)...);
        return std::move(*this);
    }
    template <typename... Components>
    Path join(Components&&... comp) const&
    {
        Path res(*this);
        res.append(std::forward<Components>(comp)...);
        return std::move(res);
    }

private:
    std::unique_ptr<std::list<Segment>> m_path_;
};

/**
 * @TODO: Python os.path.join , join("Document","/home") => "/home"  
 * 
*/
template <typename... Components>
Path join(const Path& path, Components&&... comp)
{
    Path res(path);
    res.append(std::forward<Components>(comp)...);
    return std::move(res);
}

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
inline Path operator"" _p(const char* s, std::size_t) { return Path::parse(s); }
} // namespace literals
} // namespace sp::db
#endif //SPDB_XPath_h_