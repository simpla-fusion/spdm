#include "XPath.h"
#include "../utility/Logger.h"
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
        if (pos != std::string::npos)
        {
            res = prefix.substr(0, pos) + "/" + path;
        }
        else
        {
            res = path;
        }
    }
    return res;
}

std::tuple<std::string, std::string, std::string, std::string, std::string> urlparse(std::string const& url)
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

XPath::XPath(const std::string& url)
{
    std::smatch m;
    if (!std::regex_match(url, m, url_pattern))
    {
        throw std::runtime_error("illegal request! " + url);
    }

    m_protocol_ = m[2].str();
    m_authority_ = m[4].str();
    m_query_ = m[7].str();
    m_fragment_ = m[9];

    std::string path = m[5].str();

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

XPath::XPath(const XPath& other)
    : m_protocol_(other.m_protocol_),
      m_authority_(other.m_authority_),
      m_path_(other.m_path_),
      m_query_(other.m_query_),
      m_fragment_(other.m_fragment_)
{
}

XPath::XPath(XPath&& other)
    : m_protocol_(std::move(other.m_protocol_)),
      m_authority_(std::move(other.m_authority_)),
      m_path_(std::move(other.m_path_)),
      m_query_(std::move(other.m_query_)),
      m_fragment_(std::move(other.m_fragment_))
{
}

XPath::~XPath() {}

std::string XPath::filename() const
{
    return m_path_.size() > 0 ? std::get<type_tags::Key>(m_path_.back()) : "";
}

std::string XPath::extension() const
{
    auto fname = filename();

    auto pos = fname.rfind('.');
    if (pos != std::string::npos)
    {
        return fname.substr(pos);
    }
    else
    {
        return "";
    }
}

std::string XPath::str() const
{
    NOT_IMPLEMENTED;
    return "";
}

void XPath::append(const std::string& path) { m_path_.emplace_back(path); }
void XPath::append(int idx) { m_path_.emplace_back(idx); }
void XPath::append(int b, int e, int seq) { m_path_.emplace_back(std::make_tuple(b, e, seq)); }
} // namespace sp::db