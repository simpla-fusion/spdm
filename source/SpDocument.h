#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <any>

#include "SpRange.h"

class SpXPath;
class SpDOMObject;
class SpAttribute;
class SpNode;

std::ostream &operator<<(std::ostream &os, SpDOMObject const &d);

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

class SpDOMObject
{

public:
    typedef SpDOMObject this_type;
    typedef SpRange<this_type> range;
    typedef SpRange<const this_type> const_range;

    typedef typename const_range::iterator const_iterator;
    typedef typename range::iterator iterator;

    SpDOMObject();
    SpDOMObject(SpDOMObject &&other);
    explicit SpDOMObject(SpDOMObject &parent);

    virtual ~SpDOMObject();

    // SpDOMObject(SpDOMObject const &) = delete;
    SpDOMObject &operator=(SpDOMObject const &) = delete;

    bool is_root() const { return m_parent_ == nullptr; }

    SpDOMObject *parent();
    const SpDOMObject *parent() const;

    range children();
    const_range children() const;

    range slibings();
    const_range slibings() const;

    range select(SpXPath const &path);
    const_range select(SpXPath const &path) const;

    std::ostream &repr(std::ostream &os) const;

protected:
    SpDOMObject *m_parent_ = nullptr;
};

class SpAttribute : public SpDOMObject
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

    range slibings();
    const_range slibings() const;

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

class SpNode : public SpDOMObject
{
public:
    SpNode();
    ~SpNode();

    SpNode(SpNode &&);

    SpNode(SpNode const &) = delete;
    SpNode &operator=(SpNode const &) = delete;

    bool empty() const;

    void append_child(SpNode const &);
    void append_child(SpNode &&);

    range children();
    const_range children() const;

    range select(SpXPath const &path);
    const_range select(SpXPath const &path) const;

    SpAttribute attribute(std::string const &);
    const SpAttribute attribute(std::string const &) const;

    range attributes();
    const_range attributes() const;

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

class SpDocument
{
public:
    class OID
    {
    public:
        OID();
        ~OID() = default;

        OID(unsigned long id) : m_id_(id){};

        OID(OID &&) = default;
        OID(OID const &) = default;
        OID &operator=(OID const &) = default;

        operator unsigned long() const { return m_id_; }
        unsigned long id() const { return m_id_; }

        bool operator==(OID const &other) { return m_id_ == other.m_id_; }

    private:
        unsigned long m_id_ = 0;
    };

    typedef OID id_type;

    OID oid;

    SpDocument();

    SpDocument(SpDocument &&);

    ~SpDocument();

    SpDocument(SpDocument const &) = delete;
    SpDocument &operator=(SpDocument const &) = delete;

    void schema(SpDocument const &schema);
    const SpDocument &schema();
    void schema(std::string const &schema);
    const std::string &schema_id();

    SpNode const &root() const;

    int load(std::string const &);
    int save(std::string const &);
    int load(std::istream const &);
    int save(std::ostream const &);

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

#endif //SPDB_DOCUMENT_H_