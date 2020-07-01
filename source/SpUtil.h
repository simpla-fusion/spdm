#ifndef SPDB_UTIL_H_
#define SPDB_UTIL_H_
#include <regex>

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

std::string urljoin(std::string const &prefix, std::string const &path);

#endif //SPDB_UTIL_H_