//
// Created by salmon on 18-4-12.
//

#ifndef SIMPLA_DAGRAPHUTILITY_H
#define SIMPLA_DAGRAPHUTILITY_H

#include <ostream>
namespace sp {
struct DAGraph;
std::ostream& DAGraphDrawDot(std::ostream& os, DAGraph const&, int indent = 0, int tab = 4);
}  // namespace sp{

#endif  // SIMPLA_DAGRAPHUTILITY_H
