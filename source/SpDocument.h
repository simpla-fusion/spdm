#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
#include <string>
#include <iostream>
#include <vector>
#include "SpOID.h"
class SpXPath
{
public:
    SpXPath(std::string const &);
    ~SpXPath();

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};
class SpDBAttribute
{
private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

class SpNode
{
public:
    std::vector<SpNode> select_nodes(SpXPath const &path);
    SpDBAttribute attribute(std::string const &);
    SpNode child();

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

class SpDocument
{
public:
    SpOID oid;
    SpDocument();
    ~SpDocument();

    SpNode root();

    int load(std::string const &);
    int save(std::string const &);
    int load(std::istream const &);
    int save(std::ostream const &);

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

#endif //SPDB_DOCUMENT_H_