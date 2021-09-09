//
// Created by salmon on 18-4-12.
//
#include "DAGraph.h"

#include <algorithm>  // for std::for_each
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/utility.hpp>  // for boost::tie
#include <iostream>           // for std::cout
#include <utility>            // for std::pair
namespace sp {
// DAGNode::DAGNode(std::string const& name) : spObject() {}
// DAGNode::~DAGNode() = default;

// DAGEdge::DAGEdge() : spObject(), m_ID_() {}
// DAGEdge::~DAGEdge(){};

struct DAGraphDefault : public DAGraph, SP_REGISTERED_IN_FACTORY(DAGraph, DAGraphDefault) {
    SP_REGISTERED_OBJECT_HEAD(DAGraph, DAGraphDefault)
   public:
    DAGraphDefault();
    ~DAGraphDefault() override = default;

    explicit DAGraphDefault(SpDataEntry const&);

    Status Accept(visitor_type const& visitor) const override;
    Status Deserialize(const data_entry_type& entry) override;

    bool hasCycle() const override;

    node_type* AddNode(node_type* n) override;
    edge_type* AddEdge(node_type* source_node, std::string const& key_source, node_type* target_node,
                       std::string const& key_target) override;
    node_type* CopyNode(node_type const*) override;

    // removes the node x and remove remove all edges from or to it
    Status RemoveNode(node_type const* x) override;
    // remove edge
    Status RemoveEdge(edge_type const* x) override;

    size_type NumOfNodes() const override;
    size_type NumOfEdge() const override;
    node_type const* GetNodeById(size_type id) const override;
    edge_type const* GetEdgeById(size_type id) const override;

    typedef boost::adjacency_list<> graph_t;
    graph_t m_graph_;
};

std::unique_ptr<DAGraph> NewGraph(SpDataEntry const& entry) { return std::make_unique<DAGraphDefault>(); }
DAGraphDefault::DAGraphDefault(SpDataEntry const& entry) : DAGraphDefault() { Deserialize(entry); }
DAGraphDefault::DAGraphDefault() = default;

Status DAGraphDefault::Accept(visitor_type const& visitor) const { return base_type::Accept(visitor); };
Status DAGraphDefault::Deserialize(const data_entry_type& entry) { return Set(entry); };
bool DAGraphDefault::hasCycle() const { return true; }

DAGraphDefault::node_type* DAGraphDefault::AddNode(node_type* n) {}
DAGraphDefault::edge_type* DAGraphDefault::AddEdge(node_type* source_node, std::string const& key_source,
                                                   node_type* target_node, std::string const& key_target) {}
DAGraphDefault::node_type* DAGraphDefault::CopyNode(node_type const*) {}

// removes the node x and remove remove all edges from or to it
Status DAGraphDefault::RemoveNode(node_type const* x) {}
// remove edge
Status DAGraphDefault::RemoveEdge(edge_type const* x) {}

DAGraphDefault::size_type DAGraphDefault::NumOfNodes() const {}
DAGraphDefault::size_type DAGraphDefault::NumOfEdge() const {}
DAGraphDefault::node_type const* DAGraphDefault::GetNodeById(size_type id) const {}
DAGraphDefault::edge_type const* DAGraphDefault::GetEdgeById(size_type id) const {}
//
// bool DAGraph::isAdjacent(node_type const* x, node_type const* y) { return false; }
// std::vector<DAGraph::node_type const*> DAGraph::FindNeighbors(node_type const* x) const {
//    std::vector<DAGraph::node_type const*> res;
//    return std::move(res);
//}
//
// DAGraph::node_type* DAGraph::AddNode(size_t reg_node_id) {
//    return dynamic_cast<typename DAGraph::node_type*>(m_Nodes_.Add(node_type::New(reg_node_id))->getObject());
//}
//
// DAGraph::node_type* DAGraph::CopyNode(node_type const* src) {
//    auto p = src->Copy();
//    return AddNode(p);
//}
//
// DAGraph::node_type const* DAGraph::GetNodeById(std::size_t id) const {
//    node_type const* res = nullptr;
//    for (auto const& item : m_Nodes_.container()) {
//        if (item.Find("@Id")->Equal(id)) {
//            res = dynamic_cast<node_type const*>(item.getObject());
//            break;
//        }
//    }
//    return res;
//}
// DAGraph::node_type const* DAGraph::GetNodeByName(std::string const& s_name) const {
//    node_type const* res = nullptr;
//    for (auto const& item : m_Nodes_.container()) {
//        if (item.Find("@Name")->Equal(s_name)) {
//            res = dynamic_cast<node_type const*>(item.getObject());
//            break;
//        }
//    }
//    return res;
//}

// DAGraph::edge_type* DAGraph::AddEdge(std::size_t from_id, data_entry_type output, std::size_t to_id,
//                                     data_entry_type input) {
//    auto e = new edge_type;
//    m_Edges_.Add()->Set(e);
//
//    e->m_from_node_ = GetNode(from_id);
//    e->m_FromNodeID_ = from_id;
//    e->m_FormOutput_ = std::move(output);
//    e->m_to_node_ = GetNode(to_id);
//    e->m_ToNodeID_ = to_id;
//    e->m_ToInput_ = std::move(input);
//    return e;
//}
<<<<<<< HEAD
DAGraph::edge_type* DAGraph::AddEdge(DAGOutput from_node, DAGInput to_node) {
    auto e = new edge_type;
    e->m_target_ = std::move(to_node);
    e->m_source_ = std::move(from_node);
    m_Edges_.Add()->Set(e);
    return e;
}
DAGraph::edge_type* DAGraph::AddEdge(DAGNode const* from_node, DAGNode* to_node) {
    return AddEdge(DAGOutput{from_node, ""}, DAGInput{to_node, ""});
}

// removes the edge from the vertex x to the vertex y, if it is there{}
Status DAGraph::RemoveEdge(node_type const* src, node_type const* dest) { return Status::NotModified(); }
Status DAGraph::RemoveEdge(edge_type const* x) { return Status::NotModified(); }
=======
// DAGraph::edge_type* DAGraph::AddEdge(DAGOutput from_node, DAGInput to_node) {
//    auto e = new edge_type;
//    e->m_target_ = std::move(to_node);
//    e->m_source_ = std::move(from_node);
//    m_Edges_.Add()->Set(e);
//    return e;
//}
// DAGraph::edge_type* DAGraph::AddEdge(DAGNode* from_node, DAGNode* to_node) {
//    return AddEdge(DAGOutput{from_node, ""}, DAGInput{to_node, ""});
//}
//
//// removes the edge from the vertex x to the vertex y, if it is there{}
// Status DAGraph::RemoveEdge(node_type const* src, node_type const* dest) { return Status::NotModified(); }
// Status DAGraph::RemoveEdge(edge_type const* x) { return Status::NotModified(); }
//
// struct DAGExecutorImpl : public DAGExecutor {
//    explicit DAGExecutorImpl(const DAGraph* graph) : m_graph_(graph) {}
//    ~DAGExecutorImpl() override = default;
//    void RunAsync(DoneCallback done) override;
//    const DAGraph* m_graph_;
//};
// void DAGExecutorImpl::RunAsync(sp::DAGExecutor::DoneCallback done) {
//    if (m_graph_ == nullptr) { done(Status::OK()); }
//
//    while (1) {
//        std::vector<DAGNode*> ready;
//        for (auto& p : m_graph_->m_Nodes_.container()) {
//            if (auto* node = const_cast<DAGNode*>(dynamic_cast<DAGNode const*>(p.getObject()))) {
//                if (node->isReady()) { ready.push_back(node); }
//            }
//        }
//        if (ready.empty()) {
//            done(Status::OK());
//            break;
//        } else {
//            // TODO(salmon): async run
//            for (auto* node : ready) { node->Run(); }
//
//            // TODO(salmon): push data along edges
//            //            for (auto& p : m_graph_->m_Edges_.container()) {
//            //                if (auto* edge = dynamic_cast<DAGEdge const*>(p.getObject())) {
//            //                    if (edge->Source().node->isDone()) { edge->Update(); }
//            //                }
//            //            }
//        }
//    }
//}
//
//std::unique_ptr<DAGExecutor> DAGExecutor::New(const DAGraph* graph, SpDataEntry const& data_entry) {
//    return std::make_unique<DAGExecutorDefault>(graph);
//}

>>>>>>> b135fe8dc2a6f8decb1d38c73f6b6b747456f302
}  // namespace sp