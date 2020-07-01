#ifndef SPDB_UTIL_H_
#define SPDB_UTIL_H_
#include <string>
#include <tuple>

std::string urljoin(std::string const &base, std::string const &path);


/***
 * Return:   
 *    tuple( scheme ,authority , path,query ,fragment )
 */
std::tuple<std::string, std::string, std::string, std::string, std::string> urlparser(std::string const &url);

#endif //SPDB_UTIL_H_