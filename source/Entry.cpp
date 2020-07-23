#include "Entry.h"
#include "utility/Factory.h"
namespace sp
{
std::unique_ptr<Entry> Entry::create(const std::string& uri)
{

    std::string schema = "memory";

    auto pos = uri.find(":");

    if (pos == std::string::npos)
    {
        pos = uri.rfind('.');
        if (pos != std::string::npos)
        {
            schema = uri.substr(pos);
        }
        else
        {
            schema = uri;
        }
    }
    else
    {
        schema = uri.substr(0, pos);
    }

    if (schema == "")
    {
        schema = "memory";
    }
    else if (schema == "http" || schema == "https")
    {
        NOT_IMPLEMENTED;
    }
    if (!Factory<Entry>::has_creator(schema))
    {
        RUNTIME_ERROR << "Can not parse schema " << schema << std::endl;
    }

    auto res = Factory<Entry>::create(schema);

    if (res == nullptr)
    {
        throw std::runtime_error("Can not create Entry for schema: " + schema);
    }
    else
    {
        VERBOSE << "load backend:" << schema << std::endl;
    }

    // if (schema != uri)
    // {
    //     res->fetch(uri);
    // }

    return res;
}

bool Entry::add_creator(const std::string& c_id, const std::function<Entry*()>& fun)
{
    return Factory<Entry>::add(c_id, fun);
};

} // namespace sp