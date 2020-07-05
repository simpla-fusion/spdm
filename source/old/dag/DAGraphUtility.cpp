//
// Created by salmon on 18-4-12.
//
#include "DAGraphUtility.h"
#include "DAGraph.h"
#include "SpDMIOStream.h"

namespace sp {

std::ostream& DAGraphDrawDot(std::ostream& os, DAGraph const& graph, int indent, int tab) {
    os << std::setw(indent) << " "
       << "digraph "
       << " {" << std::endl;
    os << std::setw(indent + tab) << " "
       << "subgraph cluster_" << graph.GetID() << " { " << std::endl;
    os << std::setw(indent + tab * 2) << " "
       << "label = " << graph.at("@Name") << ";" << std::endl;
//    for (auto const& p : graph.m_Nodes_.container()) {
//        auto node = dynamic_cast<DAGraph::node_type const*>(p.getObject());
//
//        os << std::setw(indent + tab * 2) << " "
//           << "node_" << node->GetID() << " [shape=record, label = ";
//        os << "\"{";
//        bool is_first = true;
//        for (auto const& input : node->GetInput().container()) {
//            if (!is_first) { os << "|"; }
//            os << "<" << input.first << ">" << input.first;
//            is_first = false;
//        }
//        os << "}";
//        os << "| " << node->GetRegisterName() << " |";
//        os << "{";
//        is_first = true;
//        for (auto const& output : node->GetOutput().container()) {
//            if (!is_first) { os << "|"; }
//            os << "<" << output.first << ">" << output.first;
//            is_first = false;
//        }
//        os << "}\"";
//        os << "];" << std::endl;
//    }
//    for (auto const& p : graph.m_Edges_.container()) {
//        auto edge = dynamic_cast<DAGraph::edge_type const*>(p.getObject());
//
//        os << std::setw(indent + tab * 2) << " "
//           << "node_" << edge->Source().node->GetID() << ":" << edge->Source().key << " -> "
//           << "node_" << edge->Target().node->GetID() << ":" << edge->Target().key << ";" << std::endl;
//    }
//    os << std::setw(indent + tab) << " "
//       << " } " << std::endl;
//
//    os << std::setw(indent) << " "
//       << "}" << std::endl;

    return os;
}

}  // namespace sp{
