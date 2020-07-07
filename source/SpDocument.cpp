#include "SpDocument.h"
using namespace sp;

SpDocument::OID::OID() : m_id_(0)
{
    // TODO:  random  init m_id_
}
SpDocument::OID::OID(unsigned long id) : m_id_(id) {}

struct SpDocument::pimpl_s
{
    SpNode *m_root_ = nullptr;
};
SpDocument::SpDocument() : m_pimpl_(new pimpl_s) {}
SpDocument::~SpDocument() { delete m_pimpl_; }
SpDocument::SpDocument(SpDocument &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ == nullptr; };

const SpNode &SpDocument::root() const { return *(m_pimpl_->m_root_); }
SpNode &SpDocument::root() { return *(m_pimpl_->m_root_); }

int SpDocument::load(const std::string &path)
{
    std::ifstream fid(path);
    int res = this->load(fid);
    return res;
}
int SpDocument::save(const std::string &path)
{
    std::ofstream fid(path);
    int res = this->save(fid);
    return res;
}
int SpDocument::load(std::istream const &) { return 0; }
int SpDocument::save(std::ostream const &) { return 0; }
