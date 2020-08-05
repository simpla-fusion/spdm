#include "Schema.h"

namespace sp::db
{
Schema::Schema() {}
Schema::~Schema() {}

bool Schema::validate(const XPath&, const Entry&) const { return true; }
} // namespace sp::db