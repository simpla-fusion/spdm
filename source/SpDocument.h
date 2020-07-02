#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <any>

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

template <typename _Tp>
class SpRange
{
public:
    class iterator : public std::iterator<std::input_iterator_tag, _Tp>
    {
    public:
        typedef iterator this_type;
        typedef std::iterator<std::input_iterator_tag, _Tp> base_type;

        using typename base_type::pointer;
        using typename base_type::reference;

        iterator();
        ~iterator();
        iterator(this_type const &);
        iterator(this_type &&);
        this_type &operator=(this_type const &);

        this_type operator++(int)
        {
            this_type res(*this);
            next();
            return res;
        }

        this_type &operator++()
        {
            next();
            return *this;
        }

        bool operator==(this_type const &other) const { return equal(other); };
        bool operator!=(this_type const &other) const { return !equal(other); }

        reference operator*() const { return *self(); }
        pointer operator->() const { return self(); }

        void next();
        bool equal(this_type const &other) const;
        size_t distance(this_type const &other) const;
        pointer self() const;

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
    };

    typedef SpRange<_Tp> this_type;

    /// One of the @link iterator_tags tag types@endlink.
    typedef typename iterator::iterator_category iterator_category;
    /// The type "pointed to" by the iterator.
    typedef typename iterator::value_type value_type;
    /// Distance between iterators is represented as this type.
    typedef typename iterator::difference_type difference_type;
    /// This type represents a pointer-to-value_type.
    typedef typename iterator::pointer pointer;
    /// This type represents a reference-to-value_type.
    typedef typename iterator::reference reference;

    SpRange() : m_b_(), m_e_(){};
    SpRange(iterator b, iterator e = iterator()) : m_b_(b), m_e_(e){};
    SpRange(this_type const &other) : m_b_(other.m_b_), m_e_(other.m_e_){};
    SpRange(this_type &&other) : m_b_(std::move(other.m_b_)), m_e_(std::move(other.m_e_)){};
    SpRange &operator=(this_type const &) = default;

    bool empty() const { return m_b_ == m_e_; }
    size_t size() const { return std::distance(m_b_, m_e_); }

    iterator begin() const { return m_b_; };
    iterator end() const { return m_e_; };

    this_type filter(SpXPath const &) const;

private:
    iterator m_b_;
    iterator m_e_;
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