#include "SpNode.h"
#include "SpEntry.h"
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

SpNode::SpNode(std::shared_ptr<SpNode> const &parent, int tag) : m_parent_(parent), m_entry_(SpEntry::create(tag)){};

SpNode::SpNode(SpNode const &other) : m_parent_(other.m_parent_), m_entry_(other.m_entry_->copy()) {}

SpNode::SpNode(SpNode &&other) : m_parent_(other.m_parent_), m_entry_(std::move(other.m_entry_)) { other.m_entry_.reset(); }

SpNode::~SpNode() {}

SpNode &SpNode::operator=(this_type const &other)
{
    this_type(other).swap(*this);
    return *this;
}

std::ostream &SpNode::repr(std::ostream &os) const { return os; }

std::ostream &operator<<(std::ostream &os, SpNode const &d) { return d.repr(os); }

void SpNode::swap(this_type &other)
{
    std::swap(m_parent_, other.m_parent_);
    std::swap(m_entry_, other.m_entry_);
}
int SpNode::type() const { return m_entry_->type(); }

bool SpNode::is_root() const { return m_parent_ == nullptr; }

bool SpNode::is_leaf() const { return !(type() == NodeTag::List || type() == NodeTag::Object); }

size_t SpNode::depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }

bool SpNode::same_as(this_type const &other) const { return this == &other; }

Attributes const &SpNode::attributes() const { return m_entry_->attributes(); }
Attributes &SpNode::attributes() { return m_entry_->attributes(); }