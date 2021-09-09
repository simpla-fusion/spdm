//
// Created by salmon on 18-4-12.
//

#include <boost/functional/hash.hpp>       //for uuid
#include <boost/uuid/uuid.hpp>             //for uuid
#include <boost/uuid/uuid_generators.hpp>  //for uuid
namespace sp {

std::size_t MakeUUID() {
    static boost::hash<boost::uuids::uuid> g_obj_hasher;
    static boost::uuids::random_generator g_uuid_generator;
    return g_obj_hasher(g_uuid_generator());
}
}  // namespace sp