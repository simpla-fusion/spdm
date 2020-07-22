#include "EntryInterface.h"
#include "utility/Factory.h"
namespace sp
{

EntryInterface::EntryInterface() : m_self_(nullptr) {}

EntryInterface::EntryInterface(const EntryInterface& other) : m_self_(other.m_self_) {}

EntryInterface::EntryInterface(EntryInterface&& other) : m_self_(other.m_self_) {}

std::unique_ptr<EntryInterface> EntryInterface::create(const std::string& uri)
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
    if (!Factory<EntryInterface>::has_creator(schema))
    {
        RUNTIME_ERROR << "Can not parse schema " << schema << std::endl;
    }

    auto res = Factory<EntryInterface>::create(schema);

    if (res == nullptr)
    {
        throw std::runtime_error("Can not create EntryInterface for schema: " + schema);
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

bool EntryInterface::add_creator(const std::string& c_id, const std::function<EntryInterface*()>& fun)
{
    return Factory<EntryInterface>::add(c_id, fun);
};

} // namespace sp