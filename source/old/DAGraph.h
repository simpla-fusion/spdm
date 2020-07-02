//
// Created by salmon on 18-4-12.
//

#ifndef SIMPLA_GRAPH_H
#define SIMPLA_GRAPH_H

#include "SpDM.h"
#include "SpDMFactory.h"

namespace simpla {
#define DAG_CONTROL_SLOT_NUM 0
struct DAGraph;
struct DAGNode;
struct DAGInput {
    DAGNode* node;
    std::string key;
};
struct DAGOutput {
    DAGNode* node;
    std::string key;
};

struct DAGNode : public spObject {
    SP_SUPPER_OBJECT_HEAD(DAGNode);
    //    explicit DAGNode(std::string const& name = "");
    //    ~DAGNode() override;
    template <typename... Args>
    explicit DAGNode(Args&&... args) : base_type() {
        base_type::Set(std::forward<Args>(args)...);
    }

    SP_CONST_ATTRIBUTE(std::size_t, ID) = MakeUUID();
    SP_ATTRIBUTE(std::string, Name);
    SP_ELEMENT(std::string, Description);
    SP_ELEMENT(spObject, Input);
    SP_ELEMENT(spObject, Output);

    DAGInput Input(std::string const& key) {
        m_Input_.Insert(key);
        return DAGInput{this, key};
    }
    DAGOutput Output(std::string const& key) {
        m_Output_.Insert(key);
        return DAGOutput{this, key};
    }

<<<<<<< HEAD
    virtual Status Run() const { return Status::OK(); }
=======
    virtual Status Run() {
        m_is_ready_ = false;
        std::cout << GetRegisterName() << std::endl;
        return Status::OK();
    }
    virtual bool isReady() const { return m_is_ready_; }
    virtual bool isDone() const { return !m_is_ready_; }
>>>>>>> b135fe8dc2a6f8decb1d38c73f6b6b747456f302

   private:
    friend class DAGraph;
    bool m_is_ready_ = true;
};

#define SP_NODE_INPUT_IMPL_(_NAME_, ...)                                                                  \
    void Set##_NAME_(__VA_ARGS__ _v_) { m_##_NAME_##_ = std::move(_v_); }                                 \
    int m_##_NAME_##_reg_ = m_Input_.Insert(__STRING(_NAME_))->Set(&m_##_NAME_##_, simpla::kIsReference); \
    __VA_ARGS__ m_##_NAME_##_

#define SP_NODE_OUTPUT_IMPL_(_NAME_, ...)                                                                  \
    auto const& Get##_NAME_() const { return m_##_NAME_##_; }                                              \
    int m_##_NAME_##_reg_ = m_Output_.Insert(__STRING(_NAME_))->Set(&m_##_NAME_##_, simpla::kIsReference); \
    __VA_ARGS__ m_##_NAME_##_

#define SP_NODE_INPUT(...) SP_ARGS_CYC_SHIFT(SP_NODE_INPUT_IMPL_, __VA_ARGS__)
#define SP_NODE_OUTPUT(...) SP_ARGS_CYC_SHIFT(SP_NODE_OUTPUT_IMPL_, __VA_ARGS__)

struct DAGEdge : public spObject {
    SP_OBJECT_HEAD(spObject, DAGEdge);
    typedef DAGNode node_type;
    typedef typename spObject::key_type key_type;

   private:
   public:
    friend class DAGraph;
    DAGInput m_target_;
    DAGOutput m_source_;

    //    DAGEdge();
    //    ~DAGEdge() override;

    //    SP_CONST_ATTRIBUTE(std::size_t, ID) = MakeUUID();

    this_type* Copy() const override { return new this_type(*this); };

    std::string GetRegisterName() const override { return __STRING(DAGEdge); }

    Status Accept(visitor_type const &visitor) const override;
    Status Deserialize(const data_entry_type &entry) override;

    auto const& Source() const { return m_source_; }
    auto const& Target() const { return m_target_; }
    int Update() { return m_target_.node->m_Input_.Set(m_target_.key, m_source_.node->m_Output_.at(m_source_.key)); }
    bool isControlEdge() const { return m_target_.key.empty(); }
};
/**
 * DAG (directed acyclic graph)
 * @quota In mathematics and computer science, a directed acyclic graph (DAG /ˈdæɡ/ (About this sound listen)), is a
 * finite  directed graph with no directed cycles. That is, it consists of finitely many vertices and edges, with
 * each edge directed from one vertex to another, such that there is no way to start at any vertex v and follow a
 * consistently-directed sequence of edges that eventually loops back to v again. Equivalently, a DAG is a directed
 * graph that has a topological ordering, a sequence of the vertices such that every edge is directed from earlier to
 * later in the sequence. -- https://en.wikipedia.org/wiki/Directed_acyclic_graph#Definitions
 */
struct DAGraph : public spObject {
    SP_SUPPER_OBJECT_HEAD(DAGraph);

    template <typename... Args>
    explicit DAGraph(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    typedef DAGNode node_type;
    typedef DAGEdge edge_type;
    typedef typename spObject::key_type key_type;

   public:
    DAGraph() = default;

    ~DAGraph() override = default;

    SP_CONST_ATTRIBUTE(std::size_t, ID) = MakeUUID();

<<<<<<< HEAD
    Status Accept(visitor_type const &visitor) const override;
    Status Deserialize(const data_entry_type &entry) override;
=======
    Status Accept(visitor_type const& visitor) const override = 0;
    Status Deserialize(const data_entry_type& entry) override = 0;

    virtual bool hasCycle() const = 0;
    virtual node_type* AddNode(node_type* n);
    virtual edge_type* AddEdge(node_type* source_node, std::string const& key_source, node_type* target_node,
                               std::string const& key_target) = 0;
    virtual node_type* CopyNode(node_type const*) = 0;

    // removes the node x and remove remove all edges from or to it
    virtual Status RemoveNode(node_type const* x) = 0;
    // remove edge
    virtual Status RemoveEdge(edge_type const* x) = 0;

    virtual size_type NumOfNodes() const = 0;
    virtual size_type NumOfEdge() const = 0;
    virtual node_type const* GetNodeById(size_type id) const = 0;
    virtual edge_type const* GetEdgeById(size_type id) const = 0;
>>>>>>> b135fe8dc2a6f8decb1d38c73f6b6b747456f302

    // tests whether there is an edge from the vertex x to the vertex y;
    bool isAdjacent(node_type const* x, node_type const* y);

    // lists all vertices y such that there is an edge from the  vertex x to the vertex y;
    std::vector<node_type const*> FindNeighbors(node_type const* x) const;

    // adds the node  as  node_desc/node_register_id

    template <typename U, typename = std::enable_if_t<std::is_base_of<node_type, U>::value>>
    node_type* AddNode(U* n) {
        return AddNode(dynamic_cast<node_type*>(n));
    };
    template <typename... Args>
    auto AddNode(Args&&... args) {
        return AddNode(node_type::New(std::forward<Args>(args)...));
    };
    template <typename U, typename... Args>
    node_type* NewNode(Args&&... args) {
        return AddNode(new U(std::forward<Args>(args)...));
    };

    // adds the edge from the vertex x to the vertex y, if it is not  there;

    edge_type* AddEdge(DAGOutput from_node, DAGInput to_node);

    // removes the edge from the vertex x to the vertex y, if it is there;
    Status RemoveEdge(node_type const* src, node_type const* dest);
};
<<<<<<< HEAD
=======
std::unique_ptr<DAGraph> NewGraph(SpDataEntry const& entry);

struct DAGExecutor {
    virtual ~DAGExecutor() = default;
    typedef std::function<void(const Status&)> DoneCallback;

    static std::unique_ptr<DAGExecutor> New(const DAGraph*, SpDataEntry const& data_entry = SpDataEntry{});

    virtual void RunAsync(DoneCallback done) = 0;

    // Synchronous wrapper for RunAsync().
    template <typename... Args>
    int Run(const Args&&... args) {
        Notification n;
        RunAsync([&n](const Status& s) { n.Notify(); }, std::forward<Args>(args)...);
        n.WaitForNotification();
        return Status::OK();
    }
};

>>>>>>> b135fe8dc2a6f8decb1d38c73f6b6b747456f302
}  // namespace simpla
#endif  // SIMPLA_GRAPH_H
