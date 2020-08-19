#include "XPath.h"
#include "../utility/Logger.h"
#include "../utility/TypeTraits.h"
#include <regex>
#include <string>

namespace sp::db
{

/**
 * https://www.ietf.org/rfc/rfc3986.txt
 * 
 *    scheme    = $2
 *    authority = $4
 *    path      = $5
 *    query     = $7
 *    fragment  = $9
 * 
 * 
*/
static const std::regex url_pattern("(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\\?([^#]*))?(#(.*))?");

static const std::regex xpath_pattern("([a-zA-Z_\\$][^/#\\[\\]]*)(\\[([^\\[\\]]*)\\])?");

std::string urljoin(std::string const& prefix, std::string const& path)
{
    std::string res;
    if (path[0] == '/')
    {
        res = path;
    }
    else
    {
        auto pos = prefix.rfind('/');

        res = (pos == std::string::npos) ? path : prefix.substr(0, pos) + "/" + path;
    }
    return res;
}

std::tuple<std::string, std::string, std::string, std::string, std::string>
urlparse(std::string const& url)
{
    std::smatch m;

    if (!std::regex_match(url, m, url_pattern))
    {
        throw std::runtime_error("illegal request! " + url);
    }

    std::string scheme = m[2].str();
    std::string authority = m[4].str();
    std::string path = m[5].str();
    std::string query = m[7].str();
    std::string fragment = m[9];
    return std::make_tuple(scheme, authority, path, query, fragment);
}

Path::Path(const std::string& path)
{
    int pos = 0;

    do
    {
        auto pend = path.find("/", pos);

        if (pend == std::string::npos)
        {
            append(path.substr(pos));
            pos = pend;
        }
        else
        {
            append(path.substr(pos, pend - pos));
            pos = pend + 1;
        }
    } while (pos != std::string::npos);
}

Path::Path(const Path& other) : m_path_(other.m_path_) {}

Path::Path(Path&& other) : m_path_(std::move(other.m_path_)) {}

std::string Path::filename() const
{
    return m_path_.size() > 0 ? std::get<segment_tags::Key>(m_path_.back()) : "";
}

std::string Path::extension() const
{
    auto fname = filename();
    auto pos = fname.rfind('.');

    return (pos == std::string::npos) ? "" : fname.substr(pos);
}

std::string Path::str() const
{
    std::ostringstream os;

    for (auto&& item : m_path_)
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::nullptr_t) {},
                [&](const std::string& seg) { os << "/" << seg; },
                [&](int idx) { os << "[" << idx << "]"; },
                [&](const std::tuple<int, int, int>& slice) { //
                    os << "["
                       << std::get<0>(slice) << ","
                       << std::get<1>(slice) << ","
                       << std::get<2>(slice)
                       << "]";
                } //
            },
            item);
    }

    return os.str();
}

Path Path::join(const Path& other) const
{
    Path res(*this);
    res.m_path_.insert(m_path_.end(), other.m_path_.begin(), other.m_path_.end());
    return std::move(res);
}

//--------------------------------------------------------------------------------

URI::URI(const std::string& uri)
{
    std::smatch m;
    if (!std::regex_match(uri, m, url_pattern))
    {
        throw std::runtime_error("illegal request! " + uri);
    }

    m_protocol_ = m[2].str();
    m_authority_ = m[4].str();
    m_query_ = m[7].str();
    m_fragment_ = m[9];

    Path(m[5].str()).swap(m_path_);
}
} // namespace sp::db