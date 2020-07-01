#include "SpDBUtil.h"
#include "pugixml/pugixml.hpp"

#include <string>

std::string path_append(std::string const &prefix, std::string const &path)
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
