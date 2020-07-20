#include "EntryInterface.h"
#include "utility/Factory.h"
namespace sp
{

EntryInterface::EntryInterface() : m_self_(nullptr) {}

EntryInterface::EntryInterface(const EntryInterface& other) : m_self_(other.m_self_) {}

EntryInterface::EntryInterface(EntryInterface&& other) : m_self_(other.m_self_) {}

void EntryInterface::bind(Entry* self) { m_self_ = self; }

std::unique_ptr<EntryInterface> EntryInterface::create(const std::string& rpath)
{

    auto pos = rpath.find(":");

    if (pos != std::string::npos)
    {
        auto p = Factory<EntryInterface>::create(rpath.substr(0, pos));

        p->fetch(rpath);

        return p;
    }
    else
    {
        return Factory<EntryInterface>::create(rpath != "" ? rpath : "memory");
    }
}
bool EntryInterface::add_creator(const std::string& c_id, const std::function<EntryInterface*()>& fun)
{
    return Factory<EntryInterface>::add(c_id, fun);
};

} // namespace sp