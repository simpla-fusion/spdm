#include "SpUtil.h"
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

    std::string urljoin(std::string const &prefix, std::string const &path)
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
    std::tuple<std::string, std::string, std::string, std::string, std::string> urlparse(std::string const &url)
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
} // namespace sp