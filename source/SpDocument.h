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

    class iterator : public std::iterator<
                         std::input_iterator_tag, // iterator_category
                         SpAttribute,             // value_type
                         long,                    // difference_type
                         SpAttribute *,           // pointer
                         SpAttribute &            // reference
                         >
    {
    public:
        explicit iterator(value_type &&);
        iterator &operator++();
        iterator operator++(int);
        bool operator==(iterator other) const;
        bool operator!=(iterator other) const;
        reference operator*() const;
        pointer operator->() const;

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
    };

    class const_iterator : public std::iterator<
                               std::input_iterator_tag, // iterator_category
                               const SpAttribute,       // value_type
                               long,                    // difference_type
                               SpAttribute const *,     // pointer
                               SpAttribute const &      // reference
                               >
    {
    public:
        explicit const_iterator(value_type &&);
        const_iterator &operator++();
        const_iterator operator++(int);
        bool operator==(const_iterator other) const;
        bool operator!=(const_iterator other) const;
        reference operator*() const;
        pointer operator->() const;

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
    };

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

    SpNode clone();

    class iterator : public std::iterator<
                         std::input_iterator_tag, // iterator_category
                         SpNode,                  // value_type
                         long,                    // difference_type
                         SpNode *,                // pointer
                         SpNode &                 // reference
                         >
    {
    public:
        explicit iterator(value_type &&);
        iterator &operator++();
        iterator operator++(int);
        bool operator==(iterator other) const;
        bool operator!=(iterator other) const;
        reference operator*() const;
        pointer operator->() const;

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
    };

    class const_iterator : public std::iterator<
                               std::input_iterator_tag, // iterator_category
                               const SpNode,            // value_type
                               long,                    // difference_type
                               SpNode const *,          // pointer
                               SpNode const &           // reference
                               >
    {
    public:
        explicit const_iterator(value_type &&);
        const_iterator &operator++();
        const_iterator operator++(int);
        bool operator==(const_iterator other) const;
        bool operator!=(const_iterator other) const;
        reference operator*() const;
        pointer operator->() const;

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
    };

    bool empty() const;
    SpNode parent() const;
    SpNode child() const;
    void append_child(SpNode const &);
    void append_child(SpNode &&);

    std::any get_attribute(std::string const &) const;
    void set_attribute(std::string const &, std::any);

    std::pair<iterator, iterator> children();
    std::pair<const_iterator, const_iterator> children() const;

    std::pair<iterator, iterator> select(SpXPath const &path = "");
    std::pair<const_iterator, const_iterator> select(SpXPath const &path = "") const;

    std::pair<SpAttribute::iterator, SpAttribute::iterator> SpNode::attributes() {}
    std::pair<SpAttribute::const_iterator, SpAttribute::const_iterator> SpNode::attributes() const {}

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

class SpDocument
{
public:
    SpOID oid;
    SpDocument();

    SpDocument(SpDocument &&);

    ~SpDocument();

    SpDocument(SpDocument const &) = delete;
    SpDocument &operator=(SpDocument const &) = delete;

    void schema(SpDocument const &schema);
    const SpDocument &schema();
    void schema(std::string const &schema);
    const std::string &schema_id();

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