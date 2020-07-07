#ifndef SP_UTIL_H_
#define SP_UTIL_H_
#include <boost/format.hpp>
#include <exception>
#include <functional>
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

 
    class NotImplementedException : public std::logic_error
    {
    public:
        NotImplementedException(std::string const &prefix = "") : std::logic_error{prefix + " Function  not yet implemented."} {}
    };
} // namespace sp

#if __GNUC__
#define NOT_IMPLEMENTED                                                                                                     \
    {                                                                                                                       \
        throw sp::NotImplementedException((boost::format("[%s:%d][%s]") % __FILE__ % __LINE__ % __PRETTY_FUNCTION__).str()); \
    }
#else
#define NOT_IMPLEMENTED                                                             \
    {                                                                               \
        throw sp::NotImplementedException("[" __FILE__ ":" __LINE__ "]:" __FUNC__); \
    }
#endif
#endif //SP_UTIL_H_