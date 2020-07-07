#ifndef SP_XPATH_H_
#define SP_XPATH_H_

#include <algorithm>
#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>
namespace sp
{
    //##############################################################################################################
    class XPath
    {
    public:
        XPath(std::string const &path = "");
        XPath(const char *path);
        ~XPath() = default;

        XPath(XPath &&) = default;
        XPath(XPath const &) = default;
        XPath &operator=(XPath const &) = default;

        const std::string &str() const;

        XPath operator/(std::string const &suffix) const;
        operator std::string() const;

    private:
        std::string m_path_;
    };

} // namespace sp
#endif //SP_XPATH_H_