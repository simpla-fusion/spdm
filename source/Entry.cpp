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

std::string to_string(Entry::element_t const& s)
{
    std::ostringstream os;
    switch (s.index())
    {
    case Entry::ElementType::String: // std::string
        os << std::get<0>(s);
        break;
    case Entry::ElementType::Boolean: // bool
        os << std::boolalpha << std::get<1>(s);
        break;
    case Entry::ElementType::Integer: // int
        os << std::get<2>(s);
        break;
    case Entry::ElementType::Float: // double
        os << std::get<3>(s);
        break;
    case Entry::ElementType::Complex: // std::complex<4>
        os << std::get<4>(s);
        break;
    case Entry::ElementType::IntVec3: //   std::array<int, 3>,
        os << std::get<5>(s)[0] << "," << std::get<5>(s)[1] << "," << std::get<5>(s)[2];
        break;
    case Entry::ElementType::FloatVec3: //   std::array<int, 3>,
        os << std::get<6>(s)[0] << "," << std::get<6>(s)[1] << "," << std::get<6>(s)[2];
        break;

    default:
        break;
    }
    return os.str();
}

Entry::element_t from_string(const std::string& s, int idx)
{
    Entry::element_t res;

    switch (idx)
    {
    case Entry::ElementType::String: // std::string
        res.emplace<std::string>(s);
        break;
    case Entry::ElementType::Boolean: // bool
        res.emplace<bool>(s == "true" || s == "True");
        break;
    case Entry::ElementType::Integer: // int
        res.emplace<long>(std::stol(s));
        break;
    case Entry::ElementType::Float: // double
        res.emplace<double>(std::stod(s));
        break;
    case Entry::ElementType::Complex: // std::complex<4>
    case Entry::ElementType::IntVec3: //   std::array<int, 3>,
    case Entry::ElementType::FloatVec3: //   std::array<int, 3>,

    default:
        NOT_IMPLEMENTED;
        break;
    }
    return res;
}
} // namespace sp