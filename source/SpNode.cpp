#include "SpNode.h"
#include "SpUtil.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace sp;

//#########################################################################################################
SpXPath::SpXPath(const std::string &path) : m_path_(path) {}
// SpXPath::~SpXPath() = default;
// SpXPath::SpXPath(SpXPath &&) = default;
// SpXPath::SpXPath(SpXPath const &) = default;
// SpXPath &SpXPath::operator=(SpXPath const &) = default;
const std::string &SpXPath::str() const { return m_path_; }

SpXPath SpXPath::operator/(const std::string &suffix) const { return SpXPath(urljoin(m_path_, suffix)); }
SpXPath::operator std::string() const { return m_path_; }

//----------------------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------------------

// SpNode::iterator SpNode::first_child() const
// {
//     NOT_IMPLEMENTED;
//     return iterator();
// }

// SpNode::range SpNode::children() const
// {
//     NOT_IMPLEMENTED;
//     return SpNode::range();
// }

// SpNode::range SpNode::select(SpXPath const &selector) const
// {
//     NOT_IMPLEMENTED;
//     return SpNode::range();
// }

// std::shared_ptr<SpNode> SpNode::select_one(SpXPath const &selector) const
//     NOT_IMPLEMENTED;
// {
//     return nullptr;
// }

// std::shared_ptr<SpNode> SpNode::child(std::string const &key) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<SpNode> SpNode::child(std::string const &key)
// {

//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<SpNode> SpNode::child(int idx)
// {

//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<SpNode> SpNode::child(int idx) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// // SpNode SpNode::insert_before(int idx)
// // {
// //     return SpNode(m_entry_->insert_before(idx));
// // }

// // SpNode SpNode::insert_after(int idx)
// // {
// //     return SpNode(m_entry_->insert_after(idx));
// // }

// int SpNode::remove_child(int idx)
// {
//     NOT_IMPLEMENTED;
//     return 0;
// }

// int SpNode::remove_child(std::string const &key)
// {
//     NOT_IMPLEMENTED;
//     return 0;
// }

// //----------------------------------------------------------------------------------------------------------
// // level 2

// ptrdiff_t SpNode::distance(this_type const &target) const { return path(target).size(); }

// SpNode::range SpNode::ancestor() const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// SpNode::range SpNode::descendants() const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// SpNode::range SpNode::leaves() const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// SpNode::range SpNode::slibings() const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// SpNode::range SpNode::path(SpNode const &target) const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// //----------------------------------------------------------------------------------------------------------
// // Content
// //----------------------------------------------------------------------------------------------------------

// class SpContent
// {
// public:
//     std::unique_ptr<SpContent> copy() const { return std::unique_ptr<SpContent>(new SpContent(*this)); };
//     NodeTag type() const { return NodeTag::Null; }; //
// };

struct node_tag_in_memory
{
};

node_t *create_node(std::shared_ptr<node_t> const &parent, NodeTag tag)
{
    return new SpNode<node_tag_in_memory>(parent);
};
