#include "SpDocument.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "SpUtil.h"
#include "SpDocBackend.h"

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

std::ostream &operator<<(std::ostream &os, SpDOMObject const &d)
{

    os << "NOT IMPLEMENTED!" << std::endl;
    return d.repr(os);
};
//#########################################################################################################

template <>
struct SpRange<SpDOMObject>::iterator::pimpl_s
{
};
template <>
SpRange<SpDOMObject>::iterator::iterator() : m_pimpl_(new pimpl_s)
{
}
template <>
SpRange<SpDOMObject>::iterator::iterator(this_type const &other)
{
}
template <>
SpRange<SpDOMObject>::iterator::iterator(this_type &&other) : m_pimpl_(other.m_pimpl_)
{
    other.m_pimpl_ = nullptr;
}
template <>
SpRange<SpDOMObject>::iterator::~iterator()
{
    delete m_pimpl_;
}
template <>
void SpRange<SpDOMObject>::iterator ::next(){};
template <>
bool SpRange<SpDOMObject>::iterator::equal(this_type const &other) const { return true; }
template <>
size_t SpRange<SpDOMObject>::iterator::distance(this_type const &other) const { return 0; }
template <>
typename SpRange<SpDOMObject>::iterator::pointer SpRange<SpDOMObject>::iterator::self() const { return nullptr; }
//----------------------------------------------------------------------------------------------------------
template <>
struct SpRange<const SpDOMObject>::iterator::pimpl_s
{
};
template <>
SpRange<const SpDOMObject>::iterator::iterator() : m_pimpl_(new pimpl_s)
{
}
template <>
SpRange<const SpDOMObject>::iterator::iterator(this_type const &other)
{
}
template <>
SpRange<const SpDOMObject>::iterator::iterator(this_type &&other) : m_pimpl_(other.m_pimpl_)
{
    other.m_pimpl_ = nullptr;
}
template <>
SpRange<const SpDOMObject>::iterator::~iterator()
{
    delete m_pimpl_;
}
template <>
void SpRange<const SpDOMObject>::iterator ::next(){};
template <>
bool SpRange<const SpDOMObject>::iterator::equal(this_type const &other) const { return true; }
template <>
size_t SpRange<const SpDOMObject>::iterator::distance(this_type const &other) const { return 0; }
template <>
typename SpRange<const SpDOMObject>::iterator::pointer SpRange<const SpDOMObject>::iterator::self() const { return nullptr; }

//#########################################################################################################
SpDOMObject::SpDOMObject() : m_parent_(nullptr) {}
SpDOMObject::~SpDOMObject() {}
SpDOMObject::SpDOMObject(SpDOMObject &parent) : m_parent_(&parent) {}

SpDOMObject *SpDOMObject::parent()
{
    if (m_parent_ == nullptr)
    {
        throw std::runtime_error("parent node is null!");
    }
    return m_parent_;
}
const SpDOMObject *SpDOMObject::parent() const
{
    if (m_parent_ == nullptr)
    {
        throw std::runtime_error("parent node is null!");
    }
    return m_parent_;
}

SpDOMObject::range SpDOMObject::children()
{
    SpDOMObject::range range;
    return std::move(range);
}
SpDOMObject::const_range SpDOMObject::children() const
{
    SpDOMObject::const_range range;
    return std::move(range);
}
SpDOMObject::range SpDOMObject::slibings()
{
    return this->parent()->children();
}
SpDOMObject::const_range SpDOMObject::slibings() const
{
    return this->parent()->children();
}
SpDOMObject::range SpDOMObject::select(SpXPath const &path)
{
    SpDOMObject::range r;
    return std::move(r);
}
SpDOMObject::const_range SpDOMObject::select(SpXPath const &path) const
{
    SpDOMObject::const_range r;
    return std::move(r);
}

std::ostream &SpDOMObject::repr(std::ostream &os) const
{
    return os;
}

class SpAttribute::pimpl_s
{
};

SpAttribute::SpAttribute() : m_pimpl_(new pimpl_s) { ; }
SpAttribute::~SpAttribute() { delete m_pimpl_; }
SpAttribute::SpAttribute(SpAttribute &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; }

SpAttribute::range SpAttribute::slibings()
{
    return dynamic_cast<SpNode *>(parent())->attributes();
}
SpAttribute::const_range SpAttribute::slibings() const
{
    return dynamic_cast<const SpNode *>(parent())->attributes();
}
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

typename SpNode::range SpNode::select(SpXPath const &path)
{
    SpNode::range nodes;
    return std::move(nodes);
}

typename SpNode::const_range SpNode::select(SpXPath const &path) const
{
    SpNode::const_range nodes;
    return std::move(nodes);
}

typename SpNode::range SpNode::children()
{
    SpNode::range nodes;
    return std::move(nodes);
}
typename SpNode::const_range SpNode::children() const
{
    SpNode::const_range nodes;
    return std::move(nodes);
}

SpAttribute SpNode::attribute(std::string const &)
{
    SpAttribute attr;
    return std::move(attr);
}

typename SpNode::range SpNode::attributes()
{
    range attr;
    return std::move(attr);
}
typename SpNode::const_range SpNode::attributes() const
{
    const_range attrs;
    return std::move(attrs);
}

SpDocument::OID::OID() : m_id_(0)
{
    // TODO:  random  init m_id_
}

struct SpDocument::pimpl_s
{
    SpNode m_root_;
};

SpDocument::SpDocument() : m_pimpl_(new pimpl_s) {}
SpDocument::~SpDocument() { delete m_pimpl_; }
SpDocument::SpDocument(SpDocument &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; };
SpNode const &SpDocument::root() const { return m_pimpl_->m_root_; }

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
