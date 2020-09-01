#include "Node.h"
#include <string>

namespace sp::db
{
Node fetch_op(const Node& object, const std::string& op, const Node& opt);

Node update_op(Node& object, const std::string& op, const Node& opt);

} // namespace sp::db