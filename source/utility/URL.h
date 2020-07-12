#ifndef SP_URL_H_
#define SP_URL_H_
#include <string>
#include <tuple>

namespace sp
{
    std::string urljoin(std::string const &base, std::string const &path);
    /***
     * Return:   
     *    tuple( scheme ,authority , path,query ,fragment )
     */
    std::tuple<std::string, std::string, std::string, std::string, std::string> urlparser(std::string const &url);

} // namespace sp
#endif //SP_URL_H_