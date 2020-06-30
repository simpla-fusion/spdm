#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <any>

class SpOID
{
public:
    SpOID();
    ~SpOID() = default;

    SpOID(unsigned long id) : m_id_(id){};

    SpOID(SpOID &&) = default;
    SpOID(SpOID const &) = default;
    SpOID &operator=(SpOID const &) = default;

    operator unsigned long() const { return m_id_; }
    unsigned long id() const { return m_id_; }

    bool operator==(SpOID const &other) { return m_id_ == other.m_id_; }

private:
    unsigned long m_id_ = 0;
};

class SpXPath
{
public:
    SpXPath(std::string const &path = "");
    SpXPath(const char *path);
    ~SpXPath() = default;

    SpXPath(SpXPath &&) = default;
    SpXPath(SpXPath const &) = default;
    SpXPath &operator=(SpXPath const &) = default;

    std::string const &value() const;

    SpXPath operator/(std::string const &suffix) const;
    operator std::string() const;

private:
    std::string m_path_;
};
class SpAttribute
{
public:
    SpAttribute();
    ~SpAttribute();
    SpAttribute(SpAttribute &&other);

    SpAttribute(SpAttribute const &) = delete;
    SpAttribute &operator=(SpAttribute const &) = delete;

    std::any get() const;
    void set(std::any const &);

    template <typename T>
    void set(T const &v) { this->set(std::any(v)); };

    template <typename T>
    SpAttribute &operator=(T const &v)
    {
        this->set(v);
        return *this;
    }

    template <typename T>
    T as() const { return std::any_cast<T>(this->get()); }

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

class SpNode
{
public:
    SpNode();
    ~SpNode();

    SpNode(SpNode &&);

    SpNode(SpNode const &) = delete;
    SpNode &operator=(SpNode const &) = delete;

    class iterator
    {
        SpNode &operator*();
        const SpNode &operator*() const;
        SpNode *operator->();
        const SpNode *operator->() const;
    };
    class range
    {
        iterator begin();
        iterator end();
        const iterator begin() const;
        const iterator end() const;
    };

    SpNode child();
    const SpNode child() const;

    range children();
    const range children() const;

    range select(SpXPath const &path = "");
    const range select(SpXPath const &path = "") const;

    SpAttribute attribute(std::string const &);
    const SpAttribute attribute(std::string const &) const;

    std::map<std::string, SpAttribute> attributes();
    std::map<std::string, const SpAttribute> attributes() const;

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

    SpDocument(SpDocument &&);

    SpDocument(SpDocument const &) = delete;
    SpDocument &operator=(SpDocument const &) = delete;

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