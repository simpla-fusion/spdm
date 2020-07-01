#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <any>
#include <functional>
template <typename BaseIterator>
class filtered_iterator
{
public:
    typedef filtered_iterator<BaseIterator> this_type;

    typedef std::remove_reference_t<decltype(*std::declval<BaseIterator>())> value_type;

    typedef value_type *pointer;

    typedef value_type &reference;

    typedef BaseIterator base_iterator_type;

    typedef std::function<bool(const value_type &)> filter_type;

    filtered_iterator() = default;
    ~filtered_iterator() = default;
    filtered_iterator(filtered_iterator const &) = default;
    filtered_iterator(filtered_iterator &&) = default;

    filtered_iterator &operator=(filtered_iterator const &) = default;

    template <typename Filter>
    filtered_iterator(base_iterator_type const &b, base_iterator_type const &e, Filter const &filter = {})
        : m_begin_(b), m_end_(e), m_filter_(filter)
    {
        while (m_begin_ != m_end_ && !m_filter_(*m_begin_))
        {
            ++(m_begin_);
        }
    }

    bool operator==(this_type const &other) const { return m_begin_ == other.m_begin_; }
    bool operator!=(this_type const &other) const { return m_begin_ != other.m_begin_; }

    template <typename Other>
    bool operator==(Other const &other) const { return m_begin_ == other; }

    template <typename Other>
    bool operator!=(Other const &other) const { return m_begin_ != other; }

    reference operator*() const { return *m_begin_; }
    pointer operator->() const { return m_begin_; }

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
    void next()
    {
        ++m_begin_;
        while (m_begin_ != m_end_ && !m_filter_(*m_begin_))
        {
            ++m_begin_;
        }
    }

private:
    filter_type m_filter_;
    base_iterator_type m_begin_, m_end_;
};

template <typename _T1, typename _T2 = _T1>
class SpRange;

template <typename _T1, typename _T2>
SpRange<_T1, _T2> make_range(_T1 const &f, _T2 const &s)
{
    return SpRange<_T1, _T2>(f, s);
}
template <typename _T1, typename _T2>
class SpRange : public std::pair<_T1, _T2>
{
public:
    typedef std::pair<_T1, _T2> base_type;

    typedef SpRange<_T1, _T2> this_type;

    using base_type::first;
    using base_type::second;

    SpRange(){};
    ~SpRange(){};
    SpRange(_T1 const &b, _T2 const &e) : base_type(b, e){};
    SpRange(this_type const &other) : base_type(other){};
    SpRange(this_type &&other) : base_type(std::forward<this_type>(other)){};
    SpRange &operator=(this_type const &) = default;

    bool empty() const { return first == second || same_as(first, second); }
    size_t size() const { return distance(first, second); }

    auto &begin() const { return base_type::first; };
    auto &end() const { return base_type::second; };

    template <typename Pred>
    auto filter(Pred const &pred) const
    {
        return make_range(filtered_iterator(first, second, pred), second);
    }
};

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

    class iterator;

    typedef SpRange<iterator> range;

    SpDOMObject();

    SpDOMObject(SpDOMObject &&other);

    virtual ~SpDOMObject();

    SpDOMObject(SpDOMObject *parent);

    SpDOMObject(SpDOMObject const &) = delete;

    SpDOMObject &operator=(SpDOMObject const &) = delete;

    bool is_root() const { return m_parent_ == nullptr; }

    SpDOMObject *parent() const;

    iterator next() const;

    iterator first_child() const;

    range children() const;

    range slibings() const;

    range select(SpXPath const &path) const;

    std::ostream &repr(std::ostream &os) const;

protected:
    SpDOMObject *m_parent_ = nullptr;
};

class SpDOMObject::iterator
{
public:
    typedef iterator this_type;

    typedef SpDOMObject value_type;
    typedef value_type *pointer;
    typedef value_type &reference;

    iterator(pointer d = nullptr) : m_self_(d){};
    ~iterator() = default;
    iterator(this_type const &) = default;
    iterator(this_type &&) = default;

    this_type &operator=(this_type const &) = default;

    bool operator==(this_type const &other) const { return same_as(other); }
    bool operator!=(this_type const &other) const { return !operator==(other); }

    reference operator*() const { return *m_self_; };
    pointer operator->() const { return m_self_; };

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

    void next();
    bool same_as(this_type const &other) const;
    ptrdiff_t distance(this_type const &other) const;

private:
    pointer m_self_;
};

class SpAttribute : public SpDOMObject
{
public:
    SpAttribute();
    ~SpAttribute();
    SpAttribute(SpAttribute &&other);

    SpAttribute(SpAttribute const &) = delete;
    SpAttribute &operator=(SpAttribute const &) = delete;

    std::string name() const;
    std::any value() const;

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

    range children() const;

    range select(SpXPath const &path) const;

    SpAttribute attribute(std::string const &) const;

    range attributes() const;

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