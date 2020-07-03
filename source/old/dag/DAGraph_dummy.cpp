//
// Created by salmon on 18-4-12.
//
//
#include <spdm/DAGraph.h>
#include <spdm/SpDMIOStream.h>
#include <iostream>
#include "DAGraphUtility.h"
//
// using namespace simpla;
//
namespace simpla {
struct Foo1 : public DAGNode, SP_REGISTERED_IN_FACTORY(DAGNode, Foo1) {
    SP_REGISTERED_OBJECT_HEAD(DAGNode, Foo1);
    template <typename... Args>
    explicit Foo1(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    SP_NODE_OUTPUT(double, Mass) = 1.0;
    SP_NODE_OUTPUT(double, Z) = 1.0;
    SP_NODE_INPUT(double, q) = 1.0;

    //    template <typename... Args>
    //    explicit Foo1(Args&&... args) : base_type(std::forward<Args>(args)...) {}
};
struct Foo2 : public DAGNode, SP_REGISTERED_IN_FACTORY(DAGNode, Foo2) {
    SP_REGISTERED_OBJECT_HEAD(DAGNode, Foo2);
    //    template <typename... Args>
    //    explicit Foo2(Args&&... args) : base_type(std::forward<Args>(args)...) {}
    SP_NODE_INPUT(double, m) = 1.0;
    SP_NODE_INPUT(double, q) = 1.0;
    SP_NODE_OUTPUT(double, E) = 1.0;
    SP_NODE_OUTPUT(double, B) = 1.0;
};
}  // namespace simpla{
//
// int main(int argc, char** argv) {
//    DAGraph graph("@Name"_ = "DAG 1");
//    auto n0 = graph.NewNode<Foo1>("@Name"_ = "first", "Description"_ = "This is a test", "Z"_ = 10);
//    auto n1 = graph.NewNode<Foo2>();
//    auto e0 = graph.AddEdge(n0->Output("Mass"), n1->Input("m"));
//    auto e1 = graph.AddEdge(n0->Output("Z"), n1->Input("q"));
//
//    //    std::cout << "DAGraph" << graph << std::endl;
//    //    DAGraphDrawDot(std::cout, graph);
//
//    DAGExecutor::New(&graph)->Run();
//
//    std::cout << "The End!" << std::endl;
//}

#include <algorithm>  // for std::for_each
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/utility.hpp>  // for boost::tie
#include <iostream>           // for std::cout
#include <utility>            // for std::pair

using namespace boost;
//
// template <class Graph>
// struct exercise_vertex {
//    exercise_vertex(Graph& g_, const char name_[]) : g(g_), name(name_) {}
//    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
//    void operator()(const Vertex& v) const {
//        using namespace boost;
//        typename property_map<Graph, vertex_index_t>::type vertex_id = get(vertex_index, g);
//        std::cout << "vertex: " << name[get(vertex_id, v)] << std::endl;
//
//        // Write out the outgoing edges
//        std::cout << "\tout-edges: ";
//        typename graph_traits<Graph>::out_edge_iterator out_i, out_end;
//        typename graph_traits<Graph>::edge_descriptor e;
//        for (boost::tie(out_i, out_end) = out_edges(v, g); out_i != out_end; ++out_i) {
//            e = *out_i;
//            Vertex src = source(e, g), targ = target(e, g);
//            std::cout << "(" << name[get(vertex_id, src)] << "," << name[get(vertex_id, targ)] << ") ";
//        }
//        std::cout << std::endl;
//
//        // Write out the incoming edges
//        std::cout << "\tin-edges: ";
//        typename graph_traits<Graph>::in_edge_iterator in_i, in_end;
//        for (boost::tie(in_i, in_end) = in_edges(v, g); in_i != in_end; ++in_i) {
//            e = *in_i;
//            Vertex src = source(e, g), targ = target(e, g);
//            std::cout << "(" << name[get(vertex_id, src)] << "," << name[get(vertex_id, targ)] << ") ";
//        }
//        std::cout << std::endl;
//
//        // Write out all adjacent vertices
//        std::cout << "\tadjacent vertices: ";
//        typename graph_traits<Graph>::adjacency_iterator ai, ai_end;
//        for (boost::tie(ai, ai_end) = adjacent_vertices(v, g); ai != ai_end; ++ai)
//            std::cout << name[get(vertex_id, *ai)] << " ";
//        std::cout << std::endl;
//    }
//    Graph& g;
//    const char* name;
//};

template <class Allocator>
struct list_with_allocatorS {};

namespace boost {
template <typename V>
struct container_gen<std::list<simpla::DAGEdge*>, V> {
    typedef std::list<simpla::DAGEdge*> type;
};
template <>
struct parallel_edge_traits<std::list<simpla::DAGEdge*>> {
    typedef allow_parallel_edge_tag type;
};

template <typename V>
struct container_gen<std::vector<simpla::DAGNode*>, V> {
    typedef std::vector<simpla::DAGEdge*> type;
};
template <>
struct parallel_edge_traits<std::vector<simpla::DAGNode*>> {
    typedef allow_parallel_edge_tag type;
};
}

// now you can define a graph using std::list and a specific allocator
using namespace boost;
using namespace simpla;
int main(int, char* []) {
    // create a typedef for the Graph type
    typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, std::string,
                                  std::map<std::string, std::string>>
        Graph;

    // declare a graph object, adding the edges and edge properties

    Graph g;

    auto v1 = boost::add_vertex(g);
    g[v1] = "v1";
    auto v2 = boost::add_vertex(g);
    g[v2] = "v2";

    auto e12 = boost::add_edge(v1, v2, g).first;

    g[e12].emplace("E1", "E2");

    std::cout << "vertices(g) = ";

    for (auto const& v : g.vertex_set()) { std::cout << g[v] << std::endl; }

    std::cout << std::endl;

<<<<<<< HEAD
    DAGraphDrawDot(std::cout, graph);
=======
    //    std::cout << "edges(g) = ";
    //    graph_traits<Graph>::edge_iterator ei, ei_end;
    //    for (boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei)
    //        std::cout << "(" << name[get(vertex_id, source(*ei, g))] << "," << name[get(vertex_id, target(*ei, g))] <<
    //        ") ";
    //    std::cout << std::endl;
    //
    //    std::for_each(vertices(g).first, vertices(g).second, exercise_vertex<Graph>(g, name));
    //
    std::map<std::string, std::string> graph_attr, vertex_attr, edge_attr;
    graph_attr["size"] = "3,3";
    graph_attr["rankdir"] = "LR";
    graph_attr["ratio"] = "fill";
    vertex_attr["shape"] = "circle";

    //    boost::write_graphviz(std::cout, g, [&](auto& os, auto const& v) { os << "[ lable=\"" << g[v] << "\"]"; },
    //                          [&](auto& os, auto const& p) {});

    return 0;
>>>>>>> b135fe8dc2a6f8decb1d38c73f6b6b747456f302
}