#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
namespace sp::db
{

Entry::Entry() {}

Entry::Entry(const Entry& other) {}

Entry::Entry(Entry&& other) {}

Entry::~Entry() {}

std::shared_ptr<Entry> Entry::create(const std::string& request)
{

    std::string schema = "memory";

    auto pos = request.find(":");

    if (pos == std::string::npos)
    {
        pos = request.rfind('.');
        if (pos != std::string::npos)
        {
            schema = request.substr(pos);
        }
        else
        {
            schema = request;
        }
    }
    else
    {
        schema = request.substr(0, pos);
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

    auto res = std::shared_ptr<Entry>(Factory<Entry>::create(schema).release());

    if (res == nullptr)
    {
        throw std::runtime_error("Can not create Entry for schema: " + schema);
    }
    else
    {
        VERBOSE << "load backend:" << schema << std::endl;
    }

    // if (schema != request)
    // {
    //     res->fetch(request);
    // }

    return res;
}

bool Entry::add_creator(const std::string& c_id, const std::function<Entry*()>& fun)
{
    return Factory<Entry>::add(c_id, fun);
};

} // namespace sp::db