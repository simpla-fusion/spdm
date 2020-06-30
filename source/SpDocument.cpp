#include "SpDocument.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "SpUtil.h"
#include "SpDocBackend.h"
SpOID::SpOID() : m_id_(0)
{
    // TODO:  random  init m_id_
}

SpXPath::SpXPath(std::string const &path) : m_path_(path) {}
SpXPath::SpXPath(const char *path) : m_path_(path) {}
// SpXPath::~SpXPath() = default;
// SpXPath::SpXPath(SpXPath &&) = default;
// SpXPath::SpXPath(SpXPath const &) = default;
// SpXPath &SpXPath::operator=(SpXPath const &) = default;

std::string const &SpXPath::value() const { return m_path_; }

SpXPath SpXPath::operator/(std::string const &suffix) const
{
    return SpXPath(urljoin(m_path_, suffix));
}
SpXPath::operator std::string() const { return m_path_; }

class SpAttribute::pimpl_s
{
};

SpAttribute::SpAttribute() : m_pimpl_(new pimpl_s) { ; }
SpAttribute::~SpAttribute() { delete m_pimpl_; }
SpAttribute::SpAttribute(SpAttribute &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; }

// class SpNode::range::pimpl_s
// {
// };

// SpNode &SpNode::range::iterator::operator*() const
// {
//     SpNode res;
//     return res;
// }
// SpNode *SpNode::range::iterator::operator->() const
// {
//     SpNode res;
//     return &res;
// }

// SpNode::range::iterator SpNode::range::begin() { ; }
// SpNode::range::iterator SpNode::range::end() { ; }
class SpNode::pimpl_s
{
};
SpNode::SpNode() : m_pimpl_(new pimpl_s) {}
SpNode::~SpNode() { delete m_pimpl_; }
SpNode::SpNode(SpNode &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; }

std::pair<SpNode::iterator, SpNode::iterator> SpNode::select(SpXPath const &path)
{
    SpNode::range nodes;
    return  ;
}
std::pair<SpNode::const_iterator, SpNode::const_iterator> SpNode::select(SpXPath const &path) const
{
    SpNode::range nodes;
    return std::move(nodes);
}

std::pair<SpNode::iterator, SpNode::iterator> SpNode::children() {}
std::pair<SpNode::const_iterator, SpNode::const_iterator> SpNode::children() const {}

SpAttribute SpNode::attribute(std::string const &)
{
    SpAttribute attr;
    return std::move(attr);
}

std::pair<SpAttribute::iterator, SpAttribute::iterator> SpNode::attributes() {}
std::pair<SpAttribute::const_iterator, SpAttribute::const_iterator> SpNode::attributes() const {}

SpNode SpNode::child()
{
    SpNode node;
    return std::move(node);
}
SpNode SpNode::clone() {}

struct SpDocument::pimpl_s
{
    SpNode m_root_;
};

SpDocument::SpDocument() : m_pimpl_(new pimpl_s) {}
SpDocument::~SpDocument() { delete m_pimpl_; }
SpDocument::SpDocument(SpDocument &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; };
SpNode SpDocument::root() { return m_pimpl_->m_root_.clone(); }

int SpDocument::load(std::string const &path)
{
    std::ifstream fid(path);
    int res = this->load(fid);
    return res;
}
int SpDocument::save(std::string const &path)
{
    std::ofstream fid(path);
    int res = this->save(fid);
    return res;
}
int SpDocument::load(std::istream const &) { return 0; }
int SpDocument::save(std::ostream const &) { return 0; }
