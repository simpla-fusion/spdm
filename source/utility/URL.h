#ifndef SP_URL_H_
#define SP_URL_H_
#include <string>
#include <tuple>

namespace sp
{
std::string urljoin(std::string const& base, std::string const& path);



std::tuple<std::string /*scheme */,
           std::string /*authority */,
           std::string /*path*/,
           std::string /*query*/,
           std::string /*fragment */>
urlparser(std::string const& url);

} // namespace sp
#endif //SP_URL_H_