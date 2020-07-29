#include "Path.h"
#include "Logger.h"
#include <regex>
#include <string>
namespace sp
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

Path::Path(const std::string& url)
{
    std::smatch m;
    if (!std::regex_match(url, m, url_pattern))
    {
        throw std::runtime_error("illegal request! " + url);
    }

    m_scheme_ = m[2].str();
    m_authority_ = m[4].str();
    m_query_ = m[7].str();
    m_fragment_ = m[9];

    std::string path = m[5].str();

    int pos = 0;

    while (1)
    {
        auto pend = path.find("/", pos);

        append(path.substr(pos, pend));

        if (pend == std::string::npos)
        {
            break;
        }
        else
        {
            pos = pend + 1;
        }
    }
}

Path::Path(const Path& other)
    : m_scheme_(other.m_scheme_),
      m_authority_(other.m_authority_),
      m_path_(other.m_path_),
      m_query_(other.m_query_),
      m_fragment_(other.m_fragment_)
{
}

Path::Path(Path&& other)
    : m_scheme_(std::move(other.m_scheme_)),
      m_authority_(std::move(other.m_authority_)),
      m_path_(std::move(other.m_path_)),
      m_query_(std::move(other.m_query_)),
      m_fragment_(std::move(other.m_fragment_))
{
}

Path::~Path() {}

std::string Path::str() const
{
    NOT_IMPLEMENTED;
    return "";
}
void Path::append(const std::string& path) { NOT_IMPLEMENTED; }
void Path::append(int idx) { NOT_IMPLEMENTED; }
void Path::append(int b, int e, int seq) { NOT_IMPLEMENTED; }
} // namespace sp