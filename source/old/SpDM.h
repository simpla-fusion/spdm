//
// Created by salmon on 18-1-19.
//

#ifndef SPDM_SPDM_H
#define SPDM_SPDM_H

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <typeindex>
#include <vector>
#include "Macro.h"
#include "Status.h"
#include "TypeTraits.h"
#include "Utility.h"
#include "nTuple.h"
namespace simpla {
template <typename _C = char>
struct SpDMString;
template <typename _C = char>
struct SpDMNumber;
template <typename _C = char>
struct SpDMElement;
template <typename TKey = SpDMString<>, typename TValue = SpDMElement<>>
struct SpDMArray;
template <typename TKey = SpDMString<>, typename TValue = SpDMElement<>>
struct SpDMObject;
template <typename _C = char>
struct SpDMPath;
template <typename TObj>
struct SpDMReference;
struct SpDMSerializer;
template <typename T, typename SFINAE = void>
struct SpDMSerializerT;
namespace traits {
template <typename Dest, typename Src, typename SFINA = void>
struct TypeCast;
template <typename Dest, typename Src>
struct TypeCast<Dest, Src,
                std::enable_if_t<std::is_constructible<Dest, Src>::value && !std::is_same<Dest, Src>::value>> {
    template <typename U>
    static Dest convert(U &&src) {
        return Dest(std::forward<U>(src));
    }
};
template <typename Dest, typename Src>
struct TypeCast<Dest, Src,
                std::enable_if_t<std::is_convertible<Dest, Src>::value && !std::is_constructible<Dest, Src>::value &&
                                 !std::is_same<Dest, Src>::value>> {
    template <typename U>
    static Dest convert(U &&src) {
        return static_cast<Dest>(std::forward<U>(src));
    }
};
template <typename Dest, typename Src>
struct TypeCast<Dest, Src,
                std::enable_if_t<std::is_integral<Dest>::value && !std::is_convertible<Dest, Src>::value &&
                                 !std::is_constructible<Dest, Src>::value && !std::is_same<Dest, Src>::value>> {
    template <typename U>
    static Dest convert(U &&src) {
        return static_cast<Dest>(static_cast<int64_t>(src));
    }
};
template <typename Dest, typename Src>
struct TypeCast<Dest, Src,
                std::enable_if_t<std::is_floating_point<Dest>::value && !std::is_convertible<Dest, Src>::value &&
                                 !std::is_constructible<Dest, Src>::value && !std::is_same<Dest, Src>::value>> {
    template <typename U>
    static Dest convert(U &&src) {
        return static_cast<Dest>(static_cast<double>(src));
    }
};

template <typename U>
struct TypeCast<U, U> {
    static U const &convert(U const &src) { return src; }
    static U &convert(U &src) { return src; }
};

template <typename Dest, typename Src>
Dest type_cast(Src const &v) {
    return TypeCast<Dest, Src>::convert(v);
};

template <typename U>
struct is_spdm : public std::false_type {};
template <typename _C>
struct is_spdm<SpDMElement<_C>> : public std::true_type {};

}  // namespace traits

enum spdm_type_tag {
    kNull /*          */ = 0x00,
    kBool /*          */ = 0x10,
    kChar /*          */ = 0x20,
    kNumber /*        */ = 0x30,
    kArray /*         */ = 0x40,
    kObject /*        */ = 0x50,

    kTrue /*          */ = kBool | 0x01,
    kFalse /*         */ = kBool | 0x02,

    kInt /*           */ = kNumber | 0x04,
    kUInt /*          */ = kNumber | 0x05,
    kInt64 /*         */ = kNumber | 0x06,
    kUInt64 /*        */ = kNumber | 0x07,
    kFloat /*         */ = kNumber | 0x08,
    kDouble /*        */ = kNumber | 0x09,

    kTypeMask /*      */ = 0x70,
    kIsReference = 0x80,
    kIsLink = kIsReference

};
enum {
    SPDM_MAX_TENSOR_DIMS = sizeof(uint64_t) / sizeof(uint8_t) - 2,
    SPDM_MAX_TENSOR_EXTENTS = std::numeric_limits<uint8_t>::max()
};
union spdm_flag_type {
    uint64_t tag;

    struct {
        uint8_t dims[SPDM_MAX_TENSOR_DIMS];
        uint8_t rank;
        uint8_t type;
    };
};
inline size_t spdm_type_num_of_elements(spdm_flag_type flag) {
    size_t res = 1;
    int rank = flag.rank;
    if (rank == 1) {
        res = flag.dims[0] + (((flag.type & kTypeMask) == kChar) ? 1 : 0);
    } else if (rank > 1) {
        res = 1;
        for (int n = 0; n < rank; ++n) { res *= flag.dims[n]; }
    }
    return res;
}
inline size_t spdm_type_size_in_byte(spdm_flag_type flag) {
    size_t res = 0;
    switch (flag.type) {
        case kBool:
            res = sizeof(bool);
            break;
        case kChar:
            res = sizeof(char);
            break;
        case kInt:
            res = sizeof(int);
            break;
        case kUInt:
            res = sizeof(unsigned int);
            break;
        case kInt64:
            res = sizeof(int64_t);
            break;
        case kUInt64:
            res = sizeof(uint64_t);
            break;
        case kDouble:
            res = sizeof(double);
            break;

        case kObject:
        case kArray:
        default:
            // ERR_UNIMPLEMENTED;
            break;
    }
    res *= spdm_type_num_of_elements(flag);
    return res;
}

template <typename _C = char>
struct SpDMVisitorInterface {
    typedef SpDMVisitorInterface<_C> interface_type;
    typedef _C char_type;
    typedef std::size_t size_type;
    SpDMVisitorInterface() = default;
    virtual ~SpDMVisitorInterface() = default;

    virtual interface_type *Copy() const = 0;
    virtual Status Set() = 0;
    virtual Status Set(bool b) = 0;
    virtual Status Set(int i) = 0;
    virtual Status Set(unsigned u) = 0;
    virtual Status Set(int64_t i) = 0;
    virtual Status Set(uint64_t u) = 0;
    virtual Status Set(double d) = 0;
    virtual Status Set(char_type const *str, size_type length = 0) = 0;

    virtual Status Set(bool const *b, unsigned int rank, size_type const *dims) = 0;
    virtual Status Set(int const *i, unsigned int rank, size_type const *dims) = 0;
    virtual Status Set(unsigned const *u, unsigned int rank, size_type const *dims) = 0;
    virtual Status Set(int64_t const *i, unsigned int rank, size_type const *dims) = 0;
    virtual Status Set(uint64_t const *u, unsigned int rank, size_type const *dims) = 0;
    virtual Status Set(double const *d, unsigned int rank, size_type const *dims) = 0;

    virtual Status Null() = 0;
    virtual interface_type *Array() = 0;
    virtual interface_type *Object() = 0;
    virtual interface_type *Add() = 0;
    virtual interface_type *Insert(char_type const *, size_type len = 0) = 0;

    template <typename U, unsigned int O, unsigned int... N>
    Status Set(nTupleBasic<U, O, N...> const &v) {
        static constexpr size_type dims[] = {N...};
        return Set(v.m_data_, sizeof...(N), dims);
    };
};
template <typename Handler, typename _C = char, typename SFINAE = void>
struct SpDMProxyVisitor;
template <typename _C = char>
struct SpDMVisitor {
    typedef std::size_t size_type;
    typedef _C char_type;
    typedef SpDMVisitor<_C> this_type;
    typedef SpDMVisitorInterface<char_type> interface_type;
    interface_type *m_interface_ = nullptr;

   public:
    SpDMVisitor(this_type const &other) : m_interface_(other.m_interface_->Copy()) {}
    SpDMVisitor(this_type &&other) : m_interface_(other.m_interface_) { other.m_interface_ = nullptr; }

    template <typename U, typename std::enable_if_t<std::is_base_of<interface_type, U>::value> * = nullptr>
    explicit SpDMVisitor(U *p) : m_interface_(p) {}
    template <typename U, typename... Args,
              typename std::enable_if_t<!std::is_base_of<interface_type, traits::remove_all_t<U>>::value> * = nullptr>
    explicit SpDMVisitor(U &&v, Args &&... args)
        : m_interface_(dynamic_cast<interface_type *>(new SpDMProxyVisitor<traits::remove_all_t<U>, char_type>(
              std::forward<U>(v), std::forward<Args>(args)...))) {}
    template <typename U,
              typename std::enable_if_t<std::is_base_of<interface_type, traits::remove_all_t<U>>::value> * = nullptr>
    explicit SpDMVisitor(U &&v) : m_interface_(new traits::remove_all_t<U>(std::move(v))) {}

    ~SpDMVisitor() { delete m_interface_; }

    auto Set(std::string const &str) const { return m_interface_->Set(str.c_str(), str.size()); }
    auto Set(char_type const *c, int length = 0) const { return m_interface_->Set(c, length); }
    template <typename Args>
    auto Set(Args const &args) const {
        return m_interface_->Set(args);
    }
    template <typename V, typename I0, typename I1>
    auto Set(V const *d, I0 rank, I1 const *p_dims) const {
        size_type dims[rank];
        for (int i = 0; i < rank; ++i) { dims[i] = p_dims[i]; }
        return m_interface_->Set(d, rank, dims);
    };
    auto Null() const { return m_interface_->Null(); }
    SpDMVisitor Array() const { return SpDMVisitor(m_interface_->Array()); }
    SpDMVisitor Object() const { return SpDMVisitor(m_interface_->Object()); }
    SpDMVisitor Add() const { return SpDMVisitor(m_interface_->Add()); }
    template <typename... Args>
    SpDMVisitor Insert(Args &&... args) const {
        return SpDMVisitor(m_interface_->Insert(std::forward<Args>(args)...));
    }
    SpDMVisitor Insert(std::string key) const { return SpDMVisitor(m_interface_->Insert(key.c_str(), key.size())); }
    template <typename K>
    auto operator[](K &&key) const {
        return Insert(std::forward<K>(key));
    }
};

template <typename _C>
struct SpDMProxyVisitor<SpDMElement<_C>, _C> : public SpDMVisitorInterface<_C> {
    typedef SpDMVisitorInterface<_C> interface_type;
    typedef SpDMProxyVisitor<SpDMElement<_C>, _C> this_type;
    typedef _C char_type;
    typedef std::size_t size_type;
    typedef SpDMElement<_C> value_type;
    value_type *m_buffer_;
    explicit SpDMProxyVisitor(value_type &p) : m_buffer_(&p){};
    explicit SpDMProxyVisitor(value_type *p) : m_buffer_(p){};
    ~SpDMProxyVisitor() override = default;

    interface_type *Copy() const override { return new this_type(m_buffer_); };
    Status Set() override { return 0; };
    Status Set(bool v) override { return m_buffer_->Set(v); }
    Status Set(int v) override { return m_buffer_->Set(v); }
    Status Set(unsigned v) override { return m_buffer_->Set(v); }
    Status Set(int64_t v) override { return m_buffer_->Set(v); }
    Status Set(uint64_t v) override { return m_buffer_->Set(v); }
    Status Set(double v) override { return m_buffer_->Set(v); }
    Status Set(char_type const *str, size_type length) override { return m_buffer_->Set(str, length); }
    Status Set(bool const *v, unsigned int rank, size_type const *dims) override {
        return m_buffer_->Set(v, rank, dims);
    }
    Status Set(int const *v, unsigned int rank, size_type const *dims) override {
        return m_buffer_->Set(v, rank, dims);
    }
    Status Set(unsigned const *v, unsigned int rank, size_type const *dims) override {
        return m_buffer_->Set(v, rank, dims);
    }
    Status Set(int64_t const *v, unsigned int rank, size_type const *dims) override {
        return m_buffer_->Set(v, rank, dims);
    }
    Status Set(uint64_t const *v, unsigned int rank, size_type const *dims) override {
        return m_buffer_->Set(v, rank, dims);
    }
    Status Set(double const *v, unsigned int rank, size_type const *dims) override {
        return m_buffer_->Set(v, rank, dims);
    }

    Status Null() override { return 1; }
    interface_type *Array() override {
        if (m_buffer_ == nullptr || !m_buffer_->isArray()) { value_type(kArray).swap(*m_buffer_); }
        return new this_type(m_buffer_);
    };
    interface_type *Object() override {
        if (m_buffer_ == nullptr || !m_buffer_->isObject()) { value_type(kObject).swap(*m_buffer_); }
        return new this_type(m_buffer_);
    };
    interface_type *Add() override { return new this_type(m_buffer_->Add()); };
    interface_type *Insert(char_type const *str, size_type len) override {
        return new this_type(m_buffer_->Insert(str, len));
    }
};
template <typename _C>
struct SpDMString {
    typedef SpDMString<_C> this_type;

   public:
    typedef _C char_type;
    typedef int8_t byte_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::size_t size_type;
    typedef SpDMVisitor<char_type> visitor_type;
    typedef SpDMElement<char_type> value_type;
    typedef SpDMNumber<char_type> number_type;
    typedef SpDMString<char_type> string_type;

    enum {
        MAX_CHARS = (sizeof(uint64_t)) / sizeof(char_type),
        MAX_SIZE = MAX_CHARS - 1,
        LenPos = MAX_SIZE,
    };
    union {
        uint64_t tag;
        struct {
            uint32_t length;
            uint8_t _pad[2];
            uint8_t rank;
            uint8_t type;
        };
    } flag;
    union {
        struct {
            char_type s_str[MAX_CHARS];
        };
        char_type *l_str;
        char_type **ref_str;
        char_type **cref_str;
        char_type const *l_c_str;
        std::string *str_p;
        void *ptr;
        uint64_t data;
    };
    SpDMString() : flag{.type = kChar}, data{0} {}
    SpDMString(SpDMString const &other) : SpDMString(other.c_str(), other.size()) {}
    SpDMString(SpDMString &&other) noexcept : flag(other.flag), data(other.data) {
        other.flag.tag = 0;
        other.data = 0;
    }
    explicit SpDMString(char_type const *c, size_type len = 0, unsigned int tag = kNull) : SpDMString() {
        flag.type = static_cast<uint8_t>(kChar | tag);
        flag.length = static_cast<int32_t>(len == 0 ? strlen(c) : len);
        if (flag.length > MAX_SIZE) {
            flag.rank = 1;
            if ((flag.type & kIsReference) != 0) {
                l_c_str = c;
            } else {
                l_str = new char_type[flag.length + 1];
                memcpy(l_str, c, static_cast<std::size_t>(flag.length) * sizeof(char_type));
                l_str[flag.length] = '\0';
            }
        } else {
            flag.type &= ~kIsReference;
            flag.rank = 0;
            memcpy(s_str, c, flag.length * sizeof(char_type));
            s_str[flag.length] = '\0';
            s_str[LenPos] = MAX_SIZE - flag.length;
        }
    }
    explicit SpDMString(char_type **c, size_type len = 0, unsigned int tag = kNull) : SpDMString() {
        ref_str = c;
        if (*c != nullptr && len == 0) { len = strlen(*c); }
        flag.length = len;
        flag.type |= kIsReference;
    }
    explicit SpDMString(char_type const *cb, char_type const *ce, unsigned int tag = kNull)
        : SpDMString(cb, ce - cb, tag) {}
    explicit SpDMString(std::string const &s) : SpDMString(s.c_str(), s.size()) {}
    explicit SpDMString(std::string *s, int tag = kIsReference) : SpDMString() {
        flag.type = flag.type | tag;
        str_p = (s);
    }

    ~SpDMString() {
        if (((flag.type & kIsReference) == 0) && (flag.rank > 0)) { delete[] l_str; }
    }

    void swap(SpDMString &other) {
        std::swap(data, other.data);
        std::swap(flag, other.flag);
    }

    bool isNull() const { return size() == 0; }

    int Accept(visitor_type const &entry) const { return entry.Set(c_str(), size()); }

    char_type const *c_str() const { return begin(); }
    char_type const *begin() const {
        return ((flag.type & kIsReference) != 0) ? *cref_str : ((flag.rank == 0) ? s_str : l_c_str);
    }
    char_type const *end() const { return begin() + size(); }

    size_type size() const { return (flag.rank == 0) ? (MAX_SIZE - s_str[MAX_CHARS - 1]) : (flag.length); }

    bool Equal(this_type const &other) const {
        // TODO need refactor
        static constexpr spdm_flag_type REFERENCE_MASK{.type = kIsReference};
        if ((flag.tag & ~REFERENCE_MASK.tag) != (other.flag.tag & ~REFERENCE_MASK.tag)) { return false; }
        bool res = false;
        if (flag.rank > 0) {
            res = strcmp(c_str(), other.c_str()) == 0;
        } else {
            res = data == other.data;
        }
        return res;
    }
    bool Equal(char_type const *other) const { return strcmp(c_str(), other) == 0; }
    bool Equal(std::string const &other) const { return other == c_str(); }
    template <typename U>
    auto Equal(U const &other) const
        -> std::enable_if_t<!(std::is_same<std::string, U>::value || std::is_same<this_type, U>::value ||
                              std::is_same<char_type const *, U>::value),
                            bool> {
        return false;
    }
    bool operator==(SpDMString const &other) const { return Equal(other); }
    bool operator==(char_type const *other) const { return strcmp(c_str(), other) == 0; }
    bool operator==(std::string const &other) const { return strcmp(c_str(), other.c_str()) == 0; }

    bool Less(this_type const &right) const {
        bool res = true;
        char const *l_c = c_str();
        char const *r_c = right.c_str();
        if (l_c[0] == '@') {
            if (r_c[0] == '@') {
                res = memcmp(l_c + 1, r_c + 1, flag.length) < 0;
            } else {
                res = true;
            }
        } else if (r_c[0] == '@') {
            res = false;
        } else {
            res = memcmp(l_c, r_c, flag.length + 1) < 0;
        }

        return res;
    }
    template <typename U>
    bool operator<(U const &right) const {
        return Less(this_type(right));
    }
    template <typename U>
    auto Get(U &res) const -> std::enable_if_t<std::is_arithmetic<U>::value, U> {
        res = static_cast<U>(atof(c_str()));
        return Status::OK();
    }
    template <typename U>
    auto Get(U &res) const -> std::enable_if_t<std::is_same<U, std::string>::value, int> {
        res = std::string(c_str());
        return Status::OK();
    }
    template <typename U>
    auto Get(U &res) const
        -> std::enable_if_t<!std::is_same<U, std::string>::value && !std::is_arithmetic<U>::value, int> {
        return Status::NotModified();
    }
    template <typename U>
    U as() const {
        U res;
        Get(res);
        return std::move(res);
    }
    template <typename U>
    this_type &operator=(U &&v) {
        Set(std::forward<U>(v));
        return *this;
    }
    template <typename U>
    auto operator==(U &&u) const {
        return Equal(std::forward<U>(u));
    }
    operator bool() const { return as<bool>(); }

    template <typename U>
    auto Set(U args) -> std::enable_if_t<std::is_arithmetic<U>::value, int> {
        this_type(std::to_string(args)).swap(*this);
        return Status::OK();
    }
    Status Set(string_type other) {
        if ((flag.type & kIsReference) == 0) {
            other.swap(*this);
            return Status::OK();
        } else {
            ERR_UNIMPLEMENTED;
        }
    }
    Status Set(std::string const &str) { return Set(string_type(str.c_str(), str.size())); }
    Status Set(char_type const *c, size_type len = 0) { return Set(string_type(c, len)); }
    Status Set(char_type *const *c, size_type len = 0) { return Set(string_type(c, len)); }
};
template <typename _C>
struct SpDMNumber {
   private:
    typedef SpDMNumber<_C> this_type;

   public:
    typedef _C char_type;
    typedef int8_t byte_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::size_t size_type;
    typedef SpDMVisitor<char_type> visitor_type;
    typedef SpDMElement<_C> value_type;
    typedef SpDMNumber<_C> number_type;

    enum { SHORT_NUMBER_PADDING = (sizeof(double) - sizeof(int)) / sizeof(byte_type) };
    spdm_flag_type flag;
    union {
#ifdef SP_LITTLEENDIAN
        struct {
            byte_type i_padding[SHORT_NUMBER_PADDING];
            int i;
        };
        struct {
            byte_type u_padding[SHORT_NUMBER_PADDING];
            unsigned int u;
        };
#else
        struct {
            int int_v;
            byte_type i_padding[SHORT_NUMBER_PADDING];
        };
        struct {
            unsigned int uint_v;
            byte_type u_padding[SHORT_NUMBER_PADDING];
        };
#endif  // SPDM_LITTLEENDIAN
        int64_t i64_v;
        uint64_t u64_v;
        double double_v;

        bool *bool_p;
        int *int_p;
        unsigned int *uint_p;
        int64_t *i64_p;
        uint64_t *u64_p;
        double *double_p;
        void *ptr;
        uint64_t data;
    };
    SpDMNumber() : flag{.tag = 0}, data(0) {}
    SpDMNumber(SpDMNumber const &other) : flag(other.flag) {
        if (flag.rank == 0) {
            data = other.data;
        } else {
            size_type s = spdm_type_size_in_byte(flag);
            ptr = operator new(s);
            memcpy(ptr, other.ptr, s);
        }
    }
    SpDMNumber(SpDMNumber &&other) noexcept : flag(other.flag), data(other.data) {
        other.flag.tag = 0;
        other.data = 0;
    }

    explicit SpDMNumber(spdm_flag_type f) : flag(f), data(0) {
        if (flag.rank > 0) { ptr = operator new(spdm_type_size_in_byte(flag)); }
    }
    explicit SpDMNumber(bool v) : flag{.tag = 0}, data{0} { flag.type = v ? kTrue : kFalse; }
    explicit SpDMNumber(int v) : flag{.tag = 0}, data(0) {
        int_v = v;
        flag.type = kInt;
    }
    explicit SpDMNumber(unsigned int v) : flag{.tag = 0}, data(0) {
        uint_v = v;
        flag.type = kUInt;
    }
    explicit SpDMNumber(int64_t v) : flag{.tag = 0}, data(0) {
        i64_v = v;
        flag.type = kInt64;
    }
    explicit SpDMNumber(uint64_t v) : flag{.tag = 0}, data(0) {
        u64_v = v;
        flag.type = kUInt64;
    }
    explicit SpDMNumber(double v) : flag{.tag = 0}, data(0) {
        double_v = v;
        flag.type = kDouble;
    }
    explicit SpDMNumber(float v) : flag{.tag = 0}, data(0) {
        double_v = v;
        flag.type = kDouble;
    }
    template <typename... Args>
    explicit SpDMNumber(char const *v, Args &&... args) = delete;

    template <typename V, unsigned int OWEN, unsigned int... N,
              typename SFINAE = std::enable_if_t<(std::is_integral<V>::value || std::is_floating_point<V>::value)>>
    explicit SpDMNumber(nTupleBasic<V, OWEN, N...> *v, int tag = kIsReference) : flag{.tag = 0}, data(0) {
        unsigned int rank = sizeof...(N);
        unsigned int dims[] = {N...};
        SpDMNumber(v->m_data_, rank, dims, tag).swap(*this);
    }
    template <typename V, typename TI = size_t,
              typename SFINAE = std::enable_if_t<(!std::is_same<V, char>::value) &&
                                                 (std::is_integral<V>::value || std::is_floating_point<V>::value)>>
    explicit SpDMNumber(V *v, int tag = kIsReference) : SpDMNumber(*v) {
        flag.type = flag.type | tag;
        if ((flag.type & kIsReference) != 0) {
            ptr = reinterpret_cast<void *>(const_cast<std::remove_const_t<V> *>(v));
        }
    }

    template <typename V, typename TI = size_t,
              typename SFINAE = std::enable_if_t<(!std::is_same<V, char>::value) &&
                                                 (std::is_integral<V>::value || std::is_floating_point<V>::value)>>
    explicit SpDMNumber(V *v, unsigned int rank, TI const *dims, int tag = kNull) : SpDMNumber(*v) {
        flag.rank = static_cast<uint8_t>(rank);
        flag.type = flag.type | tag;

        for (unsigned int i = 0; i < rank; ++i) { flag.dims[i] = static_cast<uint8_t>(dims[i]); }

        if ((flag.type & kIsReference) == 0) {
            size_type s = sizeof(V);
            for (unsigned int i = 0; i < rank; ++i) {
                flag.dims[i] = static_cast<uint8_t>(dims[i]);
                s *= dims[i];
            }
            ptr = operator new(s);
            memcpy(ptr, v, s);
        } else {
            ptr = reinterpret_cast<void *>(const_cast<std::remove_const_t<V> *>(v));
        }
    }

    ~SpDMNumber() {
        if (((flag.type & kIsReference) == 0) && flag.rank > 0) { operator delete(ptr); }
    }
    void swap(SpDMNumber &other) {
        std::swap(data, other.data);
        std::swap(flag, other.flag);
    }

    static spdm_flag_type compact_flag(int) { return spdm_flag_type{.type = kInt}; }
    static spdm_flag_type compact_flag(unsigned int) { return spdm_flag_type{.type = kUInt}; }
    static spdm_flag_type compact_flag(int64_t) { return spdm_flag_type{.type = kInt64}; }
    static spdm_flag_type compact_flag(uint64_t) { return spdm_flag_type{.type = kUInt64}; }
    static spdm_flag_type compact_flag(double) { return spdm_flag_type{.type = kDouble}; }
    static spdm_flag_type compact_flag(char const *) { return spdm_flag_type{.type = kChar}; }
    static spdm_flag_type compact_flag(value_type const &v) {
        return (v.flag().type == kArray) ? compact_flag(v.asArray().container()) : v.flag();
    }

    template <typename V>
    static auto compact_flag(V const &v) -> std::enable_if_t<!std::is_arithmetic<V>::value, spdm_flag_type> {
        spdm_flag_type sub_flag{.tag = 0x0};
        sub_flag.tag = 0;
        uint8_t count = 0;
        for (auto const &item : v) {
            ++count;
            spdm_flag_type other_flag = compact_flag(item);
            if (sub_flag.tag == 0) {
                sub_flag.tag = other_flag.tag;
            } else if ((sub_flag.rank > (SPDM_MAX_TENSOR_DIMS - 1)) || sub_flag.type != other_flag.type) {
                sub_flag.tag = 0;
                sub_flag.type = kArray;
            } else if (sub_flag.rank > 0) {
                sub_flag.dims[sub_flag.rank - 1] =
                    std::max(sub_flag.dims[sub_flag.rank - 1], other_flag.dims[sub_flag.rank - 1]);
            }
        }
        spdm_flag_type res{.tag = sub_flag.tag};
        if (sub_flag.type != kArray) {
            res.dims[0] = count;
            int n = sub_flag.rank;
            for (int i = 0; i < n; ++i) { res.dims[i + 1] = sub_flag.dims[i]; }
            res.rank = sub_flag.rank + static_cast<uint8_t>(1);
        }
        return res;
    }

    template <typename U, typename I, typename V>
    static auto recursive_assign_s(U *u, I const *dims, V const &v)
        -> std::enable_if_t<std::is_arithmetic<V>::value, U *> {
        *u = static_cast<U>(v);
        ++u;
        return u;
    }
    template <typename U, typename I>
    static U *recursive_assign_s(U *u, I const *dims, value_type const &v) {
        if (v.isArray()) {
            u = recursive_assign_s(u, dims, v.asArray().container());
        } else {
            *u = v.template as<U>();
            ++u;
        }
        return u;
    }
    template <typename U, typename I, typename V>
    static U *recursive_assign_s(U *u, I const *dims, V const &v,
                                 std::enable_if_t<!std::is_arithmetic<V>::value, void> *_p = nullptr) {
        int count = 0;
        auto it = begin(v);
        for (int i = 0; i < dims[0]; ++i) {
            if (it != end(v)) {
                u = recursive_assign_s(u, dims + 1, *it);
                ++it;
            } else {
#if SP_ARRAY_INITIALIZE_VALUE == SP_SNaN
                u = recursive_assign_s(u, dims + 1, std::numeric_limits<U>::signaling_NaN());
#else
                u = recursive_assign_s(u, dims + 1, 0);
#endif
            }
        }

        return u;
    }

    template <typename V>
    void recursive_assign(V const &v) {
        if (flag.rank == 0) { return; }
        switch (flag.type) {
            case kBool:
                recursive_assign_s(bool_p, flag.dims, v);
                break;
            case kInt:
                recursive_assign_s(int_p, flag.dims, v);
                break;
            case kUInt:
                recursive_assign_s(uint_p, flag.dims, v);
                break;
            case kInt64:
                recursive_assign_s(i64_p, flag.dims, v);
                break;
            case kUInt64:
                recursive_assign_s(u64_p, flag.dims, v);
                break;
            case kDouble:
                recursive_assign_s(double_p, flag.dims, v);
                break;
            default:
                break;
        }
    }

   public:
    template <typename V, unsigned int... N>
    SpDMNumber(nTupleBasic<V, N...> const &v) : flag{.tag = 0} {}

    template <typename U>
    SpDMNumber(std::initializer_list<U> const &v) : SpDMNumber(compact_flag(v)) {
        recursive_assign(v);
    }
    template <typename U>
    SpDMNumber(std::initializer_list<std::initializer_list<U>> const &v) : SpDMNumber(compact_flag(v)) {
        recursive_assign(v);
    }
    template <typename U>
    SpDMNumber(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const &v)
        : SpDMNumber(compact_flag(v)) {
        recursive_assign(v);
    }
    template <typename U>
    SpDMNumber(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> const &v)
        : SpDMNumber(compact_flag(v)) {
        recursive_assign(v);
    }
    template <typename U>
    SpDMNumber(std::initializer_list<
               std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>> const &v)
        : SpDMNumber(compact_flag(v)) {
        recursive_assign(v);
    }
    template <typename U>
    SpDMNumber(
        std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>> const &v)
        : SpDMNumber(compact_flag(v)) {
        recursive_assign(v);
    }
    template <typename U>
    SpDMNumber(
        std::initializer_list<std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>> const &v)
        : SpDMNumber(compact_flag(v)) {
        recursive_assign(v);
    }
    template <typename U>
    SpDMNumber(
        std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>>> const &v)
        : SpDMNumber(compact_flag(v)) {
        recursive_assign(v);
    }
    template <typename U>
    SpDMNumber(
        std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>>>> const &v)
        : SpDMNumber(compact_flag(v)) {
        recursive_assign(v);
    }

   private:
    template <typename U, typename V>
    static int accept_helper(U &u, V &entry) {
        int count = 0;
        if (u.flag.rank == 0) {
            switch (u.flag.type) {
                case kTrue:
                    count += entry.Set(true);
                    break;
                case kFalse:
                    count += entry.Set(false);
                    break;
                case kInt:
                    count += entry.Set(u.int_v);
                    break;
                case kUInt:
                    count += entry.Set(u.uint_v);
                    break;
                case kInt64:
                    count += entry.Set(u.i64_v);
                    break;
                case kUInt64:
                    count += entry.Set(u.u64_v);
                    break;
                case kDouble:
                    count += entry.Set(u.double_v);
                    break;

                case kBool | kIsReference:
                    count += entry.Set(*u.bool_p);
                    break;
                case kInt | kIsReference:
                    count += entry.Set(*u.int_p);
                    break;
                case kUInt | kIsReference:
                    count += entry.Set(*u.uint_p);
                    break;
                case kInt64 | kIsReference:
                    count += entry.Set(*u.i64_p);
                    break;
                case kUInt64 | kIsReference:
                    count += entry.Set(*u.u64_p);
                    break;
                case kDouble | kIsReference:
                    count += entry.Set(*u.double_p);
                    break;
                default:
                    break;
            }
        } else {
            switch (u.flag.type & (~kIsReference)) {
                case kBool:
                case kFalse:
                case kTrue:
                    count += entry.Set(u.bool_p, u.flag.rank, u.flag.dims);
                    break;
                case kInt:
                    count += entry.Set(u.int_p, u.flag.rank, u.flag.dims);
                    break;
                case kUInt:
                    count += entry.Set(u.uint_p, u.flag.rank, u.flag.dims);
                    break;
                case kInt64:
                    count += entry.Set(u.i64_p, u.flag.rank, u.flag.dims);
                    break;
                case kUInt64:
                    count += entry.Set(u.u64_p, u.flag.rank, u.flag.dims);
                    break;
                case kDouble:
                    count += entry.Set(u.double_p, u.flag.rank, u.flag.dims);
                    break;
                default:
                    break;
            }
        }
        return count;
    }

   public:
    int Accept(visitor_type const &entry) const { return accept_helper(*this, entry); }

    template <typename U>
    auto Set(std::initializer_list<U> const &v) {
        return Set(this_type(v));
    }
    template <typename U>
    auto Set(std::initializer_list<std::initializer_list<U>> const &v) {
        return Set(this_type(v));
    }
    template <typename U>
    auto Set(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const &v) {
        return Set(this_type(v));
    }
    template <typename U>
    auto Set(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> const &v) {
        return Set(this_type(v));
    }
    template <typename U>
    auto Set(std::initializer_list<
             std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>> const &v) {
        return Set(this_type(v));
    }
    template <typename U>
    auto Set(
        std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>> const &v) {
        return Set(this_type(v));
    }
    template <typename U>
    auto Set(
        std::initializer_list<std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>> const &v) {
        return Set(this_type(v));
    }
    template <typename U>
    auto Set(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<
                 std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>>> const
                 &v) {
        return Set(this_type(v));
    }
    template <typename U>
    auto Set(
        std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>>>> const
            &v) {
        return Set(this_type(v));
    }

    auto Set(number_type other) {
        if ((flag.type & kIsReference) == 0) {
            other.swap(*this);
            return Status::OK();
        }
        auto success = Status::NotModified();

        switch (other.flag.type & (~kIsReference)) {
            case kBool:
            case kFalse:
            case kTrue:
                success = Set(other.bool_p, other.flag.rank, other.flag.dims);
                break;
            case kInt:
                success = Set(other.int_p, other.flag.rank, other.flag.dims);
                break;
            case kUInt:
                success = Set(other.uint_p, other.flag.rank, other.flag.dims);
                break;
            case kInt64:
                success = Set(other.i64_p, other.flag.rank, other.flag.dims);
                break;
            case kUInt64:
                success = Set(other.u64_p, other.flag.rank, other.flag.dims);
                break;
            case kDouble:
                success = Set(other.double_p, other.flag.rank, other.flag.dims);
                break;
            default:
                break;
        }

        return success;
    }
    auto Set() {
        flag.tag = 0;
        flag.type = kNull;
        data = 0;
        return Status::OK();
    }
    template <typename V, unsigned int OWNED, unsigned int... N>
    auto Set(nTupleBasic<V, OWNED, N...> const &v) {
        auto s = traits::product<unsigned int, N...>::value;
        return Set(v.m_data_, 1, &s);
    }
    //    auto Set(char_type const *c, unsigned int length = 0) {
    //        return Set(strcasecmp(c, c + (length == 0 ? sizeof(c) : length)));
    //    }
    template <typename V, typename I>
    auto Set(V const *v, unsigned int rank, I const *dims) {
        if ((flag.type & kIsReference) == 0) {
            this_type(v, rank, dims).swap(*this);
            return Status::OK();
        }
        auto success = Status::OK();
        size_type len = 1;
        for (int s = 0; s < rank; ++s) { len *= dims[s]; }
        len = std::min(size(), len);
        switch (flag.type & (~kIsReference)) {
            case kBool:
                for (int s = 0; s < len; ++s) { bool_p[s] = static_cast<bool>(v[s]); }
                break;
            case kInt:
                for (int s = 0; s < len; ++s) { int_p[s] = static_cast<int>(v[s]); }
                break;
            case kUInt:
                for (int s = 0; s < len; ++s) { uint_p[s] = static_cast<unsigned int>(v[s]); }
                break;
            case kInt64:
                for (int s = 0; s < len; ++s) { i64_p[s] = static_cast<int64_t>(v[s]); }
                break;
            case kUInt64:
                for (int s = 0; s < len; ++s) { u64_p[s] = static_cast<uint64_t>(v[s]); }
                break;
            case kDouble:
                for (int s = 0; s < len; ++s) { double_p[s] = static_cast<double>(v[s]); }
                break;
            default:
                success = Status::NotModified();
                break;
        }
        return success;
    }

    template <typename V>
    auto Set(V v) -> std::enable_if_t<std::is_arithmetic<V>::value, int> {
        if ((flag.type & kIsReference) == 0) {
            this_type(v).swap(*this);
            return Status::OK();
        } else {
            return Set(&v, 0, (size_type *)(nullptr));
        }
    }

    bool isTensor() const { return flag.rank > 0; }
    int GetRank() const { return flag.rank; }
    int GetDimensions(size_type *dims = nullptr) const {
        auto n = static_cast<int>(flag.rank);
        if (dims != nullptr) {
            for (int i = 0; i < n; ++i) { dims[i] = static_cast<size_type>(flag.dims[i]); }
        }
        return n;
    }
    size_type size() const { return spdm_type_num_of_elements(flag); }
    size_type element_size() const { return spdm_type_num_of_elements(flag); }
    size_type element_size_in_byte() const { return spdm_type_size_in_byte(flag); }

    bool Equal(SpDMNumber const &other) const {
        if ((flag.type & ~kIsReference) != (other.flag.type & ~kIsReference)) { return false; }
        bool res = false;
        if (flag.rank > 0) {
            res = memcmp(ptr, other.ptr, element_size_in_byte()) == 0;
        } else {
            res = data == other.data;
        }
        return res;
    }
    template <typename U>
    auto Equal(U const &other) const -> std::enable_if_t<std::is_arithmetic<U>::value, bool> {
        return as<U>() == other;
    }
    template <typename U>
    auto Equal(U const &other) const -> std::enable_if_t<!std::is_arithmetic<U>::value, bool> {
        return false;
    }
    template <typename U>
    auto Get(U &res) const -> std::enable_if_t<std::is_arithmetic<U>::value, int> {
        switch (flag.type) {
            case kTrue:
                res = static_cast<U>(true);
                break;
            case kFalse:
                res = static_cast<U>(false);
                break;
            case kInt:
                res = static_cast<U>(int_v);
                break;
            case kUInt:
                res = static_cast<U>(uint_v);
                break;
            case kInt64:
                res = static_cast<U>(i64_v);
                break;
            case kUInt64:
                res = static_cast<U>(u64_v);
                break;
            case kDouble:
                res = static_cast<U>(double_v);
                break;
            case kInt | kIsReference:
                res = static_cast<U>(*int_p);
                break;
            case kUInt | kIsReference:
                res = static_cast<U>(*uint_p);
                break;
            case kInt64 | kIsReference:
                res = static_cast<U>(*i64_p);
                break;
            case kUInt64 | kIsReference:
                res = static_cast<U>(*u64_p);
                break;
            case kDouble | kIsReference:
                res = static_cast<U>(*double_p);
                break;
            default:
                break;
        }
        return Status::OK();
    }

    template <typename U>
    auto Get(U &res) const -> std::enable_if_t<std::is_same<U, std::string>::value, int> {
        if (flag.rank > 0 && ((flag.type & kTypeMask) != kChar)) { return Status::NotModified(); }
        switch (flag.type) {
            case kInt:
                res = std::to_string(int_v);
                break;
            case kUInt:
                res = std::to_string(uint_v);
                break;
            case kInt64:
                res = std::to_string(i64_v);
                break;
            case kUInt64:
                res = std::to_string(u64_v);
                break;
            case kDouble:
                res = std::to_string(double_v);
                break;

            case kInt | kIsReference:
                res = std::to_string(*int_p);
                break;
            case kUInt | kIsReference:
                res = std::to_string(*uint_p);
                break;
            case kInt64 | kIsReference:
                res = std::to_string(*i64_p);
                break;
            case kUInt64 | kIsReference:
                res = std::to_string(*u64_p);
                break;
            case kDouble | kIsReference:
                res = std::to_string(*double_p);
                break;

            default:
                break;
        }
        return Status::OK();
    }

    template <typename V, unsigned int... N>
    int Get(nTupleView<V, N...> &res) {
        assert(flag.rank > 0);
        res.m_data_ = reinterpret_cast<V *>(ptr);
        return Status::OK();
    };
    template <typename V, unsigned int... N>
    int Get(nTupleView<const V, N...> &res) const {
        assert(flag.rank > 0);
        res.m_data_ = reinterpret_cast<V const *>(ptr);
        return Status::OK();
    };

    template <typename V, unsigned int... N>
    int Get(nTuple<V, N...> &res) const {
        if (flag.rank > 0) {
            memcpy(res.m_data_, ptr,
                   std::min(size(), static_cast<size_type>(traits::product<unsigned int, N...>::value)) * sizeof(V));
            return Status::OK();
        } else {
            return Status::NotModified();
        }
    };
    template <typename U>
    U as() const {
        U res;
        Get(res);
        return std::move(res);
    }
    template <typename U>
    U as() {
        U res;
        Get(res);
        return std::move(res);
    }
    template <typename V, unsigned int... N>
    auto Set(nTupleBasic<V, N...> const &v) {
        this_type(v).swap(*this);
        return Status::OK();
    }

    template <typename U>
    this_type &operator=(U &&u) {
        Set(std::forward<U>(u));
        return *this;
    }
    operator bool() const { return as<bool>(); }
    template <typename U>
    bool operator==(U const &u) const {
        return Equal(u);
    }

    bool Less(this_type const &right) const {
        bool res = false;
        auto const &left = *this;

        switch (left.flag.type & (~kIsReference)) {
            case kInt:
                res = left.int_v < right.int_v;
                break;
            case kUInt:
                res = left.uint_v < right.uint_v;
                break;
            case kInt64:
                res = left.i64_v < right.i64_v;
                break;
            case kUInt64:
                res = left.u64_v < right.u64_v;
                break;
            case kDouble:
                res = left.double_v < right.double_v;
                break;

            default:
                break;
        }

        return res;
    }

    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, bool>::value>>
    bool *asTensor() {
        return ((flag.type == kBool) && (flag.rank > 0)) ? bool_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, bool>::value>>
    bool const *asTensor() const {
        return ((flag.type == kBool) && (flag.rank > 0)) ? bool_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, int>::value>>
    int *asTensor() {
        return ((flag.type == kInt) && (flag.rank > 0)) ? int_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, int>::value>>
    int const *asTensor() const {
        return ((flag.type == kInt) && (flag.rank > 0)) ? int_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, unsigned int>::value>>
    unsigned int *asTensor() {
        return ((flag.type == kUInt) && (flag.rank > 0)) ? uint_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, unsigned int>::value>>
    unsigned int const *asTensor() const {
        return ((flag.type == kUInt) && (flag.rank > 0)) ? uint_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, int64_t>::value>>
    int64_t *asTensor() {
        return ((flag.type == kInt64) && (flag.rank > 0)) ? i64_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, int64_t>::value>>
    int64_t const *asTensor() const {
        return ((flag.type == kInt64) && (flag.rank > 0)) ? i64_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, uint64_t>::value>>
    uint64_t *asTensor() {
        return ((flag.type == kUInt64) && (flag.rank > 0)) ? u64_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, uint64_t>::value>>
    uint64_t const *asTensor() const {
        return ((flag.type == kUInt64) && (flag.rank > 0)) ? u64_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, double>::value>>
    double *asTensor() {
        return ((flag.type == kDouble) && (flag.rank > 0)) ? double_p : nullptr;
    };
    template <typename U, typename SFINAE = std::enable_if_t<std::is_same<U, double>::value>>
    double const *asTensor() const {
        return ((flag.type == kDouble) && (flag.rank > 0)) ? double_p : nullptr;
    };
};
template <typename _C>
struct SpDMPath {
    typedef _C char_type;

    typedef SpDMPath<_C> this_type;
    typedef SpDMString<_C> key_type;
    typedef SpDMElement<_C> value_type;
    typedef SpDMObject<key_type, value_type> object_type;
    typedef SpDMArray<key_type, value_type> array_type;
    typedef SpDMString<char_type> string_type;
    typedef std::size_t size_type;
    static constexpr char_type SEP_CHAR = '/';
    static constexpr char_type ENDC = '\0';
    key_type m_key_;
    friend value_type;
    friend object_type;
    friend array_type;

   public:
    template <typename... Args>
    explicit SpDMPath(Args &&... args) : m_key_(std::forward<Args>(args)...) {}
    template <typename I, typename SFINAE = std::enable_if_t<std::is_integral<I>::value>>
    explicit SpDMPath(I const &i) : m_key_(std::to_string(i)) {}
    SpDMPath(SpDMPath const &other) : m_key_(other.m_key_) {}
    SpDMPath(SpDMPath &&other) noexcept : m_key_(std::move(other.m_key_)) {}
    ~SpDMPath() = default;

   private:
    auto InsertImplRecursive_(char_type const *path_b, char_type const *path_e, value_type *res) const {
        if (res == nullptr) { return res; }

        while (path_b < path_e && res) {
            auto c_first = std::find(path_b, path_e, SEP_CHAR);
            res = res->Insert(key_type(path_b, c_first, kIsReference));
            path_b = c_first + 1;
        }
        return res;
    }
    template <typename TValue>
    auto FindImplRecursive_(char_type const *path_b, char_type const *path_e, TValue *res) const {
        if (res == nullptr) { return res; }

        while (path_b < path_e && res) {
            auto c_first = std::find(path_b, path_e, SEP_CHAR);
            res = res->Find(key_type(path_b, c_first, kIsReference));
            path_b = c_first + 1;
        }
        return res;
    }
    auto DeleteImplRecursive_(char_type const *path_b, char_type const *path_e, value_type *res) const {
        if (res == nullptr) { return Status::NotModified(); }
        auto success = Status::NotModified();
        while (path_b < path_e && res) {
            auto c_first = std::find(path_b, path_e, SEP_CHAR);
            if (c_first == path_e) {
                success = res->Delete(key_type(path_b, c_first, kIsReference));
                break;
            } else {
                res = res->Find(key_type(path_b, c_first, kIsReference));
            }
            path_b = c_first + 1;
        }
        return success;
    }

   public:
    template <typename TObj, typename... Args>
    auto Insert(TObj &obj, Args &&... args) const {
        auto const *path_b = m_key_.begin();
        auto const *path_e = m_key_.end();
        if (path_b[0] == SEP_CHAR) { ++path_b; }

        auto c_first = std::find(path_b, path_e, SEP_CHAR);
        if (c_first != path_e) {
            auto res = InsertImplRecursive_(c_first + 1, path_e, obj.Insert(key_type(path_b, c_first, kIsReference)));
            if (res->flag().type == kNull) { res->Set(std::forward<Args>(args)...); }
            return res;
        } else {
            return obj.Insert(m_key_, std::forward<Args>(args)...);
        }
    }
    template <typename TObj, typename... Args>
    auto InsertOrAssign(TObj &obj, Args &&... args) const {
        auto const *path_b = m_key_.begin();
        auto const *path_e = m_key_.end();
        if (path_b[0] == SEP_CHAR) { ++path_b; }

        auto c_first = std::find(path_b, path_e, SEP_CHAR);
        if (c_first != path_e) {
            auto res = InsertImplRecursive_(c_first + 1, path_e, obj.Insert(key_type(path_b, c_first, kIsReference)));
            res->Set(std::forward<Args>(args)...);
            return res;
        } else {
            return obj.InsertOrAssign(m_key_, std::forward<Args>(args)...);
        }
    }

    template <typename TObj>
    auto Find(TObj &obj) const {
        auto const *path_b = m_key_.begin();
        auto const *path_e = m_key_.end();
        if (path_b[0] == SEP_CHAR) { ++path_b; }

        auto c_first = std::find(path_b, path_e, SEP_CHAR);
        if (c_first != path_e) {
            return FindImplRecursive_(c_first + 1, path_e, obj.Find(key_type(path_b, c_first, kIsReference)));
        } else {
            return obj.Find(m_key_);
        }
    }
    template <typename TObj>
    auto Delete(TObj &obj) const {
        auto const *path_b = m_key_.begin();
        auto const *path_e = m_key_.end();
        if (path_b[0] == SEP_CHAR) { ++path_b; }

        auto c_first = std::find(path_b, path_e, SEP_CHAR);
        if (c_first != path_e) {
            return DeleteImplRecursive_(c_first + 1, path_e, obj.Find(key_type(path_b, c_first, kIsReference)));
        } else {
            return obj.Delete(m_key_);
        }
    }

    decltype(auto) str() const { return m_key_; }
};
template <typename _C>
constexpr _C SpDMPath<_C>::SEP_CHAR;
template <typename _C>
constexpr _C SpDMPath<_C>::ENDC;

template <typename TObj>
struct SpDMReference {
    typedef SpDMReference<TObj> this_type;
    typedef TObj object_type;
    typedef typename TObj::path_type path_type;
    typedef typename TObj::value_type value_type;
    typedef std::size_t size_type;
    object_type *m_obj_;
    path_type m_path_;
    template <typename... Args>
    SpDMReference(object_type *obj, Args &&... args) : m_obj_(obj), m_path_(std::forward<Args>(args)...) {}
    SpDMReference(SpDMReference const &other) : m_obj_(other.m_obj_), m_path_(other.m_path_) {}
    ~SpDMReference() = default;
    decltype(auto) operator*() const {
        if (auto p = m_path_.Insert(*m_obj_)) {
            return *p;
        } else {
            throw(std::runtime_error("Insert data failed!"));
        }
    }
    auto *operator-> () const { return m_path_.Insert(*m_obj_); }
    template <typename K>
    decltype(auto) operator[](K const &k) const {
        if (auto p = m_path_.Insert(*m_obj_)) {
            return p->get(k);
        } else {
            ERR_OUT_OF_RANGE(m_path_.str().c_str());
        }
    }

    template <typename U>
    U as() const {
        if (auto p = m_path_.Find(*m_obj_)) {
            return p->template as<U>();
        } else {
            throw(std::out_of_range(m_path_.str().c_str()));
        }
    }
    template <typename U>
    this_type &operator=(U &&value) {
        m_path_.InsertOrAssign(*m_obj_, std::forward<U>(value));
        return *this;
    }
    template <typename U>
    this_type &operator=(std::initializer_list<U> const &value) {
        m_path_.InsertOrAssign(*m_obj_, value);
        return *this;
    }
    template <typename U>
    this_type &operator=(std::initializer_list<std::initializer_list<U>> const &value) {
        m_path_.InsertOrAssign(*m_obj_, value);
        return *this;
    }
    template <typename U>
    this_type &operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const &value) {
        m_path_.InsertOrAssign(*m_obj_, value);
        return *this;
    }
    template <typename U>
    this_type &operator=(
        std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> const &value) {
        m_path_.InsertOrAssign(*m_obj_, value);
        return *this;
    }
    template <typename U>
    auto Add(U &&value) const {
        if (auto p = m_path_.Insert(*m_obj_)) {
            return p->Add(std::forward<U>(value));
        } else {
            throw(std::out_of_range(m_path_.str().c_str()));
        }
    }
    auto Add() const {
        if (auto p = m_path_.Insert(*m_obj_)) {
            return p->Add();
        } else {
            throw(std::out_of_range(m_path_.str().c_str()));
        }
    }
    size_type size() const {
        if (auto p = m_path_.Find(*m_obj_)) {
            return p->size();
        } else {
            return 0;
        }
    }
};

/**
 *  a generic model for  Hierarchical Scientific Data Tree
 *  support data type:
 *  - number : double,int,unsigned int
 *  - string,short string optimize
 *  - block data: nd-array
 *  - Object
 *  - Array
 *  - Linking data
 * support file format:
 *  - JSON/JSON-LD/XML
 *  - XDMF
 *  - HDF5
 *
 *  inspired by
 *  - RapidJSON     : https://github.com/Tencent/rapidjson
 *  - nlohmann/json : https://github.com/nlohmann/json
 *  - XDMF          : http://www.xdmf.org/index.php/Main_Page
 *
 * @tparam KeyType
 * @tparam EntityType
 * @tparam BlockType
 * @tparam ObjectType
 * @tparam ArrayType
 * @tparam AllocatorType
 * @tparam SerializerType
 */
template <typename TKey, typename TValue>
struct SpDMArray {
    typedef TKey key_type;
    typedef TValue value_type;
    typedef SpDMArray<TKey, TValue> array_type;
    typedef SpDMObject<TKey, TValue> object_type;

    typedef typename key_type::char_type char_type;
    typedef SpDMPath<char_type> path_type;
    typedef SpDMVisitor<char_type> visitor_type;

    typedef SpDMNumber<char_type> number_type;

    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    using pointer = value_type *;
    using const_pointer = value_type const *;

   private:
    typedef SpDMArray<key_type, value_type> this_type;

    std::vector<value_type> m_data_;
    friend path_type;
    friend value_type;

   public:
    SpDMArray() = default;
    virtual ~SpDMArray() = default;
    SpDMArray(SpDMArray const &other) : m_data_(other.m_data_){};
    SpDMArray(SpDMArray &&other) noexcept : m_data_(std::move(other.m_data_)){};

    SpDMArray &operator=(SpDMArray const &other) {
        this_type(other).swap(*this);
        return *this;
    };
    SpDMArray &operator=(SpDMArray &&other) noexcept {
        this_type(std::move(other)).swap(*this);
        return *this;
    };
    template <typename... Args>
    explicit SpDMArray(std::tuple<Args...> const &l) {
        Set(l);
    }
    SpDMArray(std::initializer_list<value_type> const &list) : m_data_(list) {}

    virtual void swap(this_type &other) { m_data_.swap(other.m_data_); }
    bool isNull() const { return m_data_.empty(); }
    virtual size_type size() const { return m_data_.size(); }
    virtual this_type *Clone() const { return new this_type(); }
    virtual this_type *Copy() const { return new this_type(*this); }
    virtual void DeepCopy(this_type const &other) { this_type(*this).swap(*this); }
    virtual bool Equal(this_type const &other) const {
        bool res = m_data_.size() == other.m_data_.size();
        if (res) {
            auto ib = m_data_.begin();
            auto ie = m_data_.end();
            auto jb = other.m_data_.begin();
            for (; ib != ie;) {
                res = res && (ib->Equal(*jb));
                ++ib;
                ++jb;
            }
        }

        return res;
    }
    virtual bool Equal(value_type const &other) const { return other.isArray() ? Equal(other.asArray()) : false; }
    template <typename U>
    auto Equal(U const &other)
        -> std::enable_if_t<!std::is_base_of<this_type, U>::value && !std::is_same<value_type, U>::value, bool> {
        return false;
    }
    virtual bool Less(this_type const &other) const { return false; }

    virtual Status Accept(visitor_type const &visitor) const {
        if (isNull()) {
            return visitor.Null();
        } else {
            auto v = visitor.Array();
            int count = 0;
            for (auto const &item : m_data_) { count += item.Accept(v.Add()); }
        }
        return Status::OK();
    }

    auto &container() { return m_data_; }
    auto const &container() const { return m_data_; }

    value_type *Add() {
#if __cplusplus > 201402L
        return &m_data_.emplace_back(value_type{});
#else
        m_data_.emplace_back(value_type{});
        value_type *p = &(m_data_.back());
        return p;
#endif
    }
    template <typename... Args>
    value_type *Add(Args &&... args) {
        value_type *p = Add();
        p->Set(std::forward<Args>(args)...);
        return p;
    }

   private:
    template <typename... Args>
    int set_tuple_help_(std::tuple<Args...> const &t, std::index_sequence<>) {
        return 0;
    }
    template <typename... Args, size_t I0, size_t... IDX>
    int set_tuple_help_(std::tuple<Args...> const &t, std::index_sequence<I0, IDX...>) {
        Add(std::get<I0>(t));
        return 1 + set_tuple_help_(t, std::index_sequence<IDX...>());
    }

    template <typename... Args>
    int get_tuple_helper_(std::tuple<Args...> &v, std::index_sequence<>) const {
        return 0;
    }
    template <typename... Args, size_t I0, size_t... IDX>
    int get_tuple_helper_(std::tuple<Args...> &v, std::index_sequence<I0, IDX...>) const {
        std::get<I0>(v) = m_data_[I0].template as<traits::remove_cvref_t<decltype(std::get<I0>(v))>>();
        return 1 + get_tuple_helper_(v, std::index_sequence<IDX...>());
    }

   public:
    template <typename... Args>
    auto Set(std::tuple<Args...> const &v) {
        return set_tuple_help_(v, std::index_sequence_for<Args...>()) > 0 ? Status::OK() : Status::NotModified();
    }
    template <typename... Args>
    auto Set(Args &&... args) {
        return Status::NotModified();
    }
    template <typename... Args>
    int Get(std::tuple<Args...> &res) const {
        return get_tuple_helper_(res, std::index_sequence_for<Args...>()) > 0 ? Status::OK() : Status::NotModified();
    }

    template <typename... Args>
    value_type *Insert(key_type const &k, Args &&... args) {
        return Find(k);
    }
    template <typename... Args>
    value_type *InsertOrAssign(key_type const &k, Args &&... args) {
        auto res = Find(k);
        if (res) { res->Set(value_type(std::forward<Args>(args)...)); }
        return res;
    }

    value_type *Find(size_type const &s) { return (s < size()) ? &m_data_[s] : nullptr; }
    value_type const *Find(size_type const &s) const { return (s < size()) ? &m_data_[s] : nullptr; }
    value_type *Find(key_type const &k) { return Find(k.template as<size_type>()); }
    value_type const *Find(key_type const &k) const { return Find(k.template as<size_type>()); }

    decltype(auto) get(key_type const &k) { return m_data_[k.template as<size_type>()]; }
    decltype(auto) get(key_type const &k) const { return m_data_[k.template as<size_type>()]; }
    decltype(auto) at(key_type const &k) { return m_data_.at(k.template as<size_type>()); }
    decltype(auto) at(key_type const &k) const { return m_data_.at(k.template as<size_type>()); }

    int Delete(key_type const &k) {
        auto s = k.template as<size_type>();
        if (s < size()) {
            m_data_.erase(m_data_.begin() + s);
            return Status::OK();
        } else {
            return Status::NotModified();
        }
    }
    int Delete(object_type const *k) { return Status::NotModified(); }
    auto Merge(this_type const &other) {
        m_data_.insert(m_data_.end(), other.m_data_.begin(), other.m_data_.end());
        return Status::OK();
    }
    auto Merge(this_type &&other) {
        m_data_.insert(m_data_.end(), other.m_data_.begin(), other.m_data_.end());
        return Status::OK();
    }
    auto Merge(value_type v) {
        auto success = Status::NotModified();
        switch (v.flag().type & kTypeMask) {
            case kArray:
                success = Merge(v.asArray());
                break;
            default:
                success = Add(std::move(v)) == nullptr ? Status::NotModified() : Status::OK();
        }
        return success;
    }

    value_type Number() {
        value_type res;
        spdm_flag_type t_flag = number_type::compact_flag(m_data_);
        if (((t_flag.type & kTypeMask) != kNumber) || t_flag.rank == 0) {
            for (auto &item : m_data_) { item.Number(); }
        } else {
            number_type v(t_flag);
            v.recursive_assign(m_data_);
            res.Set(std::move(v));
        }
        return std::move(res);
    }

};  // struct SpDMArray;
//*************************************************************************************************************
// @addgroup{ spObject
template <typename TKey, typename TValue>
struct SpDMObject {
#if (defined(__cplusplus) && __cplusplus >= 201402L)
    // Use transparent comparator if possible, combined with perfect forwarding
    // on find() and m_count_() calls prevents unnecessary string construction.
    using table_comparator_t = std::less<>;
#else
    using table_comparator_t = std::less<TKey>;
#endif
    typedef TKey key_type;
    typedef TValue value_type;
    typedef SpDMArray<TKey, TValue> array_type;
    typedef SpDMObject<TKey, TValue> object_type;
    typedef typename key_type::char_type char_type;
    typedef typename value_type::string_type string_type;
    typedef typename value_type::number_type number_type;
    typedef SpDMVisitor<char_type> visitor_type;
    typedef value_type data_entry_type;
    typedef SpDMVisitorInterface<char_type> visitor_interface_type;

    typedef SpDMPath<char_type> path_type;

    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    using pointer = value_type *;
    using const_pointer = value_type const *;

   private:
    typedef SpDMObject<key_type, value_type> this_type;

    std::map<key_type, value_type, table_comparator_t> m_data_;
    friend path_type;

   public:
    SpDMObject() = default;
    virtual ~SpDMObject() = default;
    SpDMObject(SpDMObject const &other) : m_data_(other.m_data_){};
    SpDMObject(SpDMObject &&other) : m_data_(std::move(other.m_data_)){};
    SpDMObject(key_type k, value_type v) { m_data_.emplace(std::move(k), std::move(v)); }
    SpDMObject(value_type v) { Set(v); }

    SpDMObject &operator=(SpDMObject const &other) {
        this_type(other).swap(*this);
        return *this;
    };
    SpDMObject &operator=(SpDMObject &&other) {
        this_type(std::move(other)).swap(*this);
        return *this;
    };
    virtual void swap(SpDMObject &other) { m_data_.swap(other.m_data_); };
    auto &data() { return m_data_; }
    auto const &data() const { return m_data_; }

    virtual std::string GetRegisterName() const { return ""; }

    virtual bool isNull() const { return m_data_.empty(); }
    virtual size_type size() const { return m_data_.size(); }
    virtual this_type *Clone() const { return new this_type(); }
    virtual this_type *Copy() const { return new this_type(*this); }
    virtual void DeepCopy(this_type const &other) { this_type(*this).swap(*this); }
    virtual bool Equal(this_type const &other) const {
        bool res = m_data_.size() == other.m_data_.size();
        if (res) {
            auto ib = m_data_.begin();
            auto ie = m_data_.end();
            auto jb = other.m_data_.begin();
            for (; ib != ie; ++ib, ++jb) { res = res && ib->first.Equal(jb->first) && ib->second.Equal(jb->second); }
        }

        return res;
        return m_data_ == other.m_data_;
    }
    bool Equal(value_type const &other) const { return other.isObject() && Equal(other.asObject()); }
    template <typename U>
    auto Equal(U const &other)
        -> std::enable_if_t<!std::is_base_of<this_type, U>::value && !std::is_same<value_type, U>::value, bool> {
        return false;
    }
    virtual bool Less(this_type const &other) const { return false; }

    auto &container() { return m_data_; }
    auto const &container() const { return m_data_; }
    template <typename U>
    int Get(U &res) const {
        return Status::NotModified();
    }

    auto Set(object_type const &other) { return Merge(other); }
    auto Set(object_type &&other) { return Merge(std::move(other)); }
    auto Set(value_type v) { return Merge(v); }

    template <typename... Args>
    auto Set(Args &&... args) {
        return Status::NotModified();
    }

    auto Merge(object_type const &other) {
        m_data_.insert(other.m_data_.begin(), other.m_data_.end());
        return Status::OK();
    }
    auto Merge(object_type &&other) {
        for (auto it = other.container().begin(); it != other.container().end(); ++it) {
            m_data_[it->first].swap(it->second);
        }
        return Status::OK();
    }
    auto Merge(value_type v) {
        auto success = Status::NotModified();
        switch (v.flag().type) {
            case kObject:
                success = Merge(std::move(*v.m_object_.ptr));
                break;
            default:
                //                InsertOrAssign("_", v);
                break;
        }
        return success;
    }

    template <typename U>
    bool operator==(U const &other) const {
        return Equal(other);
    }
    this_type &operator=(value_type value) {
        if ((value.flag().type & kTypeMask) == kObject) {
            Set(std::move(value.asObject()));
        } else {
            throw(std::runtime_error("illegal container"));
        }
        return *this;
    }

    template <typename... Args>
    value_type *Insert(key_type const &k, Args &&... args) {
        auto res = m_data_.emplace(k, value_type{});
        if (res.second) { res.first->second.Set(std::forward<Args>(args)...); }
        return &res.first->second;
    }
    template <typename... Args>
    value_type *InsertOrAssign(key_type const &k, Args &&... args) {
        auto res = m_data_.emplace(k, value_type{});
        res.first->second.Set(std::forward<Args>(args)...);
        return &res.first->second;
    }
    value_type *Find(key_type const &k) {
        value_type *res = nullptr;
        auto it = m_data_.find(k);
        if (it != m_data_.end()) { res = &it->second; }
        return res;
    }
    value_type const *Find(key_type const &k) const {
        value_type const *res = nullptr;
        auto it = m_data_.find(k);
        if (it != m_data_.end()) { res = &it->second; }
        return res;
    }
    int Delete(key_type const &k) { return m_data_.erase(k) > 0 ? Status::OK() : Status::NotModified(); }

    template <typename K>
    size_type size(K const &k) const {
        if (auto p = Find(k)) {
            return p->size();
        } else {
            return 0;
        }
    }
    template <typename K, typename... Args>
    auto Insert(K const &k, Args &&... args) {
        return path_type(k).Insert(*this, std::forward<Args>(args)...);
    }
    template <typename K, typename... Args>
    auto InsertOrAssign(K const &k, Args &&... args) {
        return path_type(k).InsertOrAssign(*this, std::forward<Args>(args)...);
    }

    template <typename K>
    auto Find(K const &k) const {
        return path_type(k).Find(*this);
    }
    template <typename K>
    auto Find(K const &k) {
        return path_type(k).Find(*this);
    }
    template <typename K>
    decltype(auto) at(K const &k) const {
        if (auto p = Find(k)) {
            return *p;
        } else {
            throw(std::out_of_range(traits::to_string(k)));
        }
    }
    template <typename K>
    decltype(auto) at(K const &k) {
        if (auto p = Find(k)) {
            return *p;
        } else {
            throw(std::out_of_range(traits::to_string(k)));
        }
    }
    template <typename K>
    auto get(K const &k) {
        return SpDMReference<this_type>{this, (k)};
    }
    template <typename U, typename K>
    U as(K &&k) const {
        return at(std::forward<K>(k)).template as<U>();
    }
    template <typename U, typename K>
    U as(K &&k, U default_value) const {
        return at(std::forward<K>(k)).template as<U>(default_value);
    }
    template <typename K>
    decltype(auto) operator[](K &&k) {
        return get(std::forward<K>(k));
    }
    template <typename K>
    decltype(auto) operator[](K &&k) const {
        return at(std::forward<K>(k));
    }
    template <typename K>
    int Delete(K &&k) {
        return path_type(std::forward<K>(k)).Delete(*this);
    }

    virtual Status Accept(visitor_type const &visitor) const {
        //        if (isNull()) {
        //            return visitor.Null();
        //        } else
        {
            auto v = visitor.Object();
            if (!GetRegisterName().empty()) { v["@type"].Set(GetRegisterName()); }
            int count = 0;
            for (auto const &item : m_data_) {
                count += item.second.Accept(v.Insert(item.first.c_str(), item.first.size()));
            }
        }
        return Status::OK();
    }
    virtual Status Deserialize(const data_entry_type &entry) { return Set(entry); };

    template <typename VISITOR>
    auto Accept(VISITOR &&visitor,
                std::enable_if_t<!std::is_base_of<visitor_type, traits::remove_cvref_t<VISITOR>>::value> *sfinae =
                    nullptr) const {
        return Accept(visitor_type(visitor));
    }
    auto Serialize(visitor_type const &entry) const { return Accept(entry); };

    template <typename U>
    auto Serialize(U &&serializer) const {
        return Accept(std::forward<U>(serializer));
    }
};  // struct SpDMObject

#define SP_OBJECT_HEAD(_BASE_, ...)         \
   private:                                 \
    typedef _BASE_ base_type;               \
    typedef __VA_ARGS__ this_type;          \
                                            \
   public:                                  \
    using typename base_type::visitor_type; \
    using typename base_type::data_entry_type;

#define SP_PROPERTY_IMPL(_NAME_, ...)                                                                         \
    void Set##_NAME_(__VA_ARGS__ _v_) { m_##_NAME_##_ = std::move(_v_); }                                     \
    auto const &Get##_NAME_() const { return m_##_NAME_##_; }                                                 \
    int m_##_NAME_##_reg_ = object_type::Insert(__STRING(_NAME_))->Set(&m_##_NAME_##_, simpla::kIsReference); \
    __VA_ARGS__ m_##_NAME_##_

#define SP_ATTRIBUTE_IMPL(_NAME_, ...)                                                                \
    void Set##_NAME_(__VA_ARGS__ _v_) { m_##_NAME_##_ = std::move(_v_); }                             \
    auto const &Get##_NAME_() const { return m_##_NAME_##_; }                                         \
    int m_##_NAME_##_reg_ =                                                                           \
        object_type::Insert(__STRING(@) __STRING(_NAME_))->Set(&m_##_NAME_##_, simpla::kIsReference); \
    __VA_ARGS__ m_##_NAME_##_

#define SP_CONST_ATTRIBUTE_IMPL(_NAME_, ...)                                                          \
    auto const &Get##_NAME_() const { return m_##_NAME_##_; }                                         \
    int m_##_NAME_##_reg_ =                                                                           \
        object_type::Insert(__STRING(@) __STRING(_NAME_))->Set(&m_##_NAME_##_, simpla::kIsReference); \
    const __VA_ARGS__ m_##_NAME_##_

#define SP_ELEMENT_IMPL(_NAME_, ...)                                                                          \
    void Set##_NAME_(__VA_ARGS__ _v_) { m_##_NAME_##_ = std::move(_v_); }                                     \
    auto const &Get##_NAME_() const { return m_##_NAME_##_; }                                                 \
    int m_##_NAME_##_reg_ = object_type::Insert(__STRING(_NAME_))->Set(&m_##_NAME_##_, simpla::kIsReference); \
    __VA_ARGS__ m_##_NAME_##_

//#define SP_PROPERTY_1(_V) SP_PROPERTY_IMPL(_V, void)
//#define SP_PROPERTY_2(_1, _V) SP_PROPERTY_IMPL(_V, _1)
//#define SP_PROPERTY_3(_1, _2, _V) SP_PROPERTY_IMPL(_V, _1, _2)
//#define SP_PROPERTY_4(_1, _2, _3, _V) SP_PROPERTY_IMPL(_V, _1, _2, _3)
//#define SP_PROPERTY_5(_1, _2, _3, _4, _V) SP_PROPERTY_IMPL(_V, _1, _2, _3, _4)
//#define SP_PROPERTY_6(_1, _2, _3, _4, _5, _V) SP_PROPERTY_IMPL(_V, _1, _2, _3, _4, _5)
//#define SP_PROPERTY_7(_1, _2, _3, _4, _5, _6, _V) SP_PROPERTY_IMPL(_V, _1, _2, _3, _4, _5, _6)
//#define SP_PROPERTY_8(_1, _2, _3, _4, _5, _6, _7, _V) SP_PROPERTY_IMPL(_V, _1, _2, _3, _4, _5, _6, _7)
//#define SP_PROPERTY_9(_1, _2, _3, _4, _5, _6, _7, _8, _V) SP_PROPERTY_IMPL(_V, _1, _2, _3, _4, _5, _6, _7, _8)

#define SP_ARGS_CYC_SHIFT_1(_FUNC_) _FUNC_()
#define SP_ARGS_CYC_SHIFT_2(_FUNC_, _V) _FUNC_(_V)
#define SP_ARGS_CYC_SHIFT_3(_FUNC_, _1, _V) _FUNC_(_V, _1)
#define SP_ARGS_CYC_SHIFT_4(_FUNC_, _1, _2, _V) _FUNC_(_V, _1, _2)
#define SP_ARGS_CYC_SHIFT_5(_FUNC_, _1, _2, _3, _V) _FUNC_(_V, _1, _2, _3)
#define SP_ARGS_CYC_SHIFT_6(_FUNC_, _1, _2, _3, _4, _V) _FUNC_(_V, _1, _2, _3, _4)
#define SP_ARGS_CYC_SHIFT_7(_FUNC_, _1, _2, _3, _4, _5, _V) _FUNC_(_V, _1, _2, _3, _4, _5)
#define SP_ARGS_CYC_SHIFT_8(_FUNC_, _1, _2, _3, _4, _5, _6, _V) _FUNC_(_V, _1, _2, _3, _4, _5, _6)
#define SP_ARGS_CYC_SHIFT_9(_FUNC_, _1, _2, _3, _4, _5, _6, _7, _V) _FUNC_(_V, _1, _2, _3, _4, _5, _6, _7)
#define SP_ARGS_CYC_SHIFT_10(_FUNC_, _1, _2, _3, _4, _5, _6, _7, _8, _V) _FUNC_(_V, _1, _2, _3, _4, _5, _6, _7, _8)

#define SP_ARGS_CYC_SHIFT(...) VFUNC(SP_ARGS_CYC_SHIFT_, __VA_ARGS__)

#define SP_PROPERTY(...) SP_ARGS_CYC_SHIFT(SP_PROPERTY_IMPL, __VA_ARGS__)

#define SP_ELEMENT(...) SP_ARGS_CYC_SHIFT(SP_ELEMENT_IMPL, __VA_ARGS__)

#define SP_ATTRIBUTE(...) SP_ARGS_CYC_SHIFT(SP_ATTRIBUTE_IMPL, __VA_ARGS__)
#define SP_CONST_ATTRIBUTE(...) SP_ARGS_CYC_SHIFT(SP_CONST_ATTRIBUTE_IMPL, __VA_ARGS__)

#define SP_PROPERTY_POINTER_IMPL(_NAME_, ...)                                                                \
    void Set##_NAME_(__VA_ARGS__ *_v_) { m_##_NAME_##_ = _v_; }                                              \
    auto const *Get##_NAME_() const { return m_##_NAME_##_; }                                                \
    auto *Get##_NAME_() { return m_##_NAME_##_; }                                                            \
    int m_##_NAME_##_reg_ = object_type::Insert(__STRING(_NAME_))->Set(m_##_NAME_##_, simpla::kIsReference); \
    __VA_ARGS__ *m_##_NAME_##_

#define SP_PROPERTY_POINTER(...) SP_ARGS_CYC_SHIFT(SP_PROPERTY_POINTER_IMPL, __VA_ARGS__)

#define SP_PROPERTY_STR(_NAME_)                                                               \
    void Set##_NAME_(char const *_v_) { object_type::InsertOrAssign(__STRING(_NAME_), _v_); } \
    auto Get##_NAME_() const { return object_type::at(__STRING(_NAME_)).template as<std::string>(); }
//*************************************************************************************************************

namespace detail {
HAS_MEMBER_FUNCTION(Set)
}  // namespace detail{}
template <typename _C>
struct SpDMElement {
    typedef _C char_type;
    typedef int8_t byte_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::size_t size_type;

    typedef SpDMElement<_C> this_type;
    typedef SpDMPath<_C> path_type;
    typedef SpDMString<_C> key_type;
    typedef SpDMElement<_C> value_type;

    typedef SpDMObject<key_type, value_type> object_type;
    typedef SpDMArray<key_type, value_type> array_type;
    typedef SpDMString<_C> string_type;
    typedef SpDMNumber<_C> number_type;
    typedef SpDMVisitor<char_type> visitor_type;
    typedef SpDMVisitorInterface<char_type> visitor_interface_type;

   private:
    friend object_type;
    friend array_type;
    friend path_type;
    friend number_type;
    friend string_type;

    union {
        string_type m_string_;
        number_type m_number_;
        struct {
            spdm_flag_type flag;
            object_type *ptr;

        } m_object_;
        struct {
            spdm_flag_type flag;
            array_type *ptr;
        } m_array_;
        struct {
            spdm_flag_type m_flag_;
            uint64_t m_data_;
        };
    };

   public:
    SpDMElement() : m_flag_{.tag = 0}, m_data_(0){};
    SpDMElement(SpDMElement &&other) noexcept : m_data_(other.m_data_), m_flag_(other.m_flag_) {
        other.m_flag_.tag = 0;
        other.m_data_ = 0;
    };
    SpDMElement(SpDMElement const &other) : m_flag_(other.m_flag_), m_data_(0) {
        switch (m_flag_.type & kTypeMask) {
            case kObject:
                m_object_.ptr = other.m_object_.ptr->Copy();
                break;
            case kArray:
                m_array_.ptr = other.m_array_.ptr->Copy();
                break;
            case kNumber:
                number_type(other.m_number_).swap(m_number_);
                break;
            case kChar:
                string_type(other.m_string_).swap(m_string_);
                break;
            default:
                break;
        }
        //        }
    };
    explicit SpDMElement(spdm_type_tag t) : m_flag_{.tag = 0}, m_data_(0) {
        spdm_flag_type f{.tag = 0};
        f.type = static_cast<uint8_t>(t);
        SpDMElement(f).swap(*this);
    }
    explicit SpDMElement(spdm_flag_type f) : m_flag_(f), m_data_(0) {
        switch (m_flag_.type & kTypeMask) {
            case kObject:
                m_object_.ptr = new object_type;
                break;
            case kArray:
                m_array_.ptr = new array_type;
                break;
            case kBool:
            case kNumber:
                number_type(f).swap(m_number_);
                break;
            case kChar:
                string_type().swap(m_string_);
                break;
            case kNull:
            default:
                break;
        }
    }
    ~SpDMElement() { release(); }
    void release() {
        switch (m_flag_.type & (kTypeMask | kIsReference)) {
            case kObject:
                delete m_object_.ptr;
                m_flag_.tag = 0;
                break;
            case kArray:
                delete m_array_.ptr;
                m_flag_.tag = 0;
                break;
            case kBool:
            case kNumber:
                m_number_.~SpDMNumber();
                break;
            case kChar:
                m_string_.~SpDMString();
                break;
            case kNull:
            default:
                break;
        }
        m_flag_.type = 0;
        m_data_ = 0;
    };

    explicit SpDMElement(object_type &&other) : SpDMElement(new object_type(std::move(other))){};
    explicit SpDMElement(array_type &&other) : SpDMElement(new array_type(std::move(other))){};
    explicit SpDMElement(object_type const &other) : SpDMElement(other.Copy()){};
    explicit SpDMElement(array_type const &other) : SpDMElement(other.Copy()){};

    template <typename U, typename std::enable_if_t<std::is_base_of<object_type, U>::value> * = nullptr>
    explicit SpDMElement(U *other, unsigned int tag = kNull) : m_flag_{.tag = 0}, m_data_(0) {
        m_object_.flag.type = kObject | tag;
        m_object_.ptr = other;
    };
    template <typename U, typename std::enable_if_t<std::is_base_of<object_type, U>::value> * = nullptr>
    explicit SpDMElement(U const *other) : m_flag_{.tag = 0}, m_data_(0) {
        m_object_.flag.type = kObject;
        m_object_.ptr = other->Copy();
    };
    template <typename U, typename std::enable_if_t<std::is_base_of<array_type, U>::value> * = nullptr>
    explicit SpDMElement(U *other, unsigned int tag = kNull) : m_flag_{.tag = 0}, m_data_(0) {
        m_array_.flag.type = kArray | tag;
        m_array_.ptr = other;
    };
    template <typename U, typename std::enable_if_t<std::is_base_of<array_type, U>::value> * = nullptr>
    explicit SpDMElement(U const *other) : m_flag_{.tag = 0}, m_data_(0) {
        m_array_.flag.type = kArray;
        m_array_.ptr = other->Copy();
    };

    template <
        typename... Args,
        typename std::enable_if_t<std::is_constructible<number_type, std::remove_cv_t<Args>...>::value> * = nullptr>
    explicit SpDMElement(Args &&... args) : m_number_(std::forward<Args>(args)...){};
    template <
        typename... Args,
        typename std::enable_if_t<std::is_constructible<string_type, std::remove_cv_t<Args>...>::value> * = nullptr>
    explicit SpDMElement(Args &&... args) : m_string_(std::forward<Args>(args)...){};

    template <
        typename... Args,
        typename std::enable_if_t<std::is_constructible<array_type, std::remove_cv_t<Args>...>::value> * = nullptr>
    explicit SpDMElement(Args &&... args) : SpDMElement(new array_type(std::forward<Args>(args)...), kNull){};
    template <
        typename... Args,
        typename std::enable_if_t<std::is_constructible<object_type, std::remove_cv_t<Args>...>::value> * = nullptr>
    explicit SpDMElement(Args &&... args) : SpDMElement(new object_type(std::forward<Args>(args)...), kNull){};

    virtual this_type *Copy() const { return new this_type(*this); }

   public:
    template <typename U, typename SFINAE = std::enable_if_t<std::is_arithmetic<U>::value>>
    SpDMElement(std::initializer_list<U> const &v) : m_number_(v) {}
    template <typename U, typename SFINAE = std::enable_if_t<std::is_arithmetic<U>::value>>
    SpDMElement(std::initializer_list<std::initializer_list<U>> const &v) : m_number_(v) {}
    template <typename U, typename SFINAE = std::enable_if_t<std::is_arithmetic<U>::value>>
    SpDMElement(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const &v) : m_number_(v) {}
    template <typename U, typename SFINAE = std::enable_if_t<std::is_arithmetic<U>::value>>
    SpDMElement(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> const &v)
        : m_number_(v) {}
    template <typename U, typename SFINAE = std::enable_if_t<std::is_arithmetic<U>::value>>
    SpDMElement(std::initializer_list<
                std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>> const &v)
        : m_number_(v) {}
    template <typename U, typename SFINAE = std::enable_if_t<std::is_arithmetic<U>::value>>
    SpDMElement(
        std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>> const &v)
        : m_number_(v) {}
    template <typename U, typename SFINAE = std::enable_if_t<std::is_arithmetic<U>::value>>
    SpDMElement(
        std::initializer_list<std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>> const &v)
        : m_number_(v) {}
    template <typename U, typename SFINAE = std::enable_if_t<std::is_arithmetic<U>::value>>
    SpDMElement(
        std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>>> const &v)
        : m_number_(v) {}
    template <typename U, typename SFINAE = std::enable_if_t<std::is_arithmetic<U>::value>>
    SpDMElement(
        std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<
            std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>>>>>> const &v)
        : m_number_(v) {}

    void swap(this_type &other) {
        std::swap(m_data_, other.m_data_);
        std::swap(m_flag_, other.m_flag_);
    }
    void clear() { value_type().swap(*this); }

    void Flush() {}

    size_type size() const {
        if (m_flag_.rank > 1) { return 1; }
        size_type res = 1;
        switch (m_flag_.type) {
            case kObject:
                res = m_object_.ptr->size();
                break;
            case kArray:
                res = m_array_.ptr->size();
                break;
            case kChar:
            case kNull:
            case kTrue:
            case kFalse:
            case kInt:
            case kUInt:
            case kInt64:
            case kUInt64:
            case kFloat:
            case kDouble:
            default:
                break;
        }
        return res;
    }

    bool Equal(value_type const &other) const {
        bool res = m_flag_.tag == other.m_flag_.tag;
        if (!res) { return res; }
        switch (m_flag_.type & kTypeMask) {
            case kObject:
                res = m_object_.ptr->Equal(*other.m_object_.ptr);
                break;
            case kArray:
                res = m_array_.ptr->Equal(*other.m_array_.ptr);
                break;
            case kNumber:
                res = m_number_.Equal(other.m_number_);
                break;
            case kChar:
                res = m_string_.Equal(other.m_string_);
                break;

            default:
                break;
        }
        return res;
    }
    //    bool Equal(object_type const &other) const { return isObject() && (m_object_.ptr->Equal(other)); }
    //    bool Equal(array_type const &other) const { return isArray() && (m_array_.ptr->Equal(other)); }
    template <typename U>
    bool Equal(U const &other) const {
        bool res = false;
        switch (m_flag_.type & kTypeMask) {
            case kObject:
                res = m_object_.ptr->Equal(other);
                break;
            case kArray:
                res = m_array_.ptr->Equal(other);
                break;
            case kNumber:
                res = m_number_.Equal(other);
                break;
            case kChar:
                res = m_string_.Equal(other);
                break;
            default:
                break;
        }
        return res;
    }

    bool operator==(value_type const &other) const { return Equal(other); }
    template <typename U>
    bool operator==(U const &other) const {
        return Equal(other);
    }
    template <typename U>
    bool operator!=(U const &other) const {
        return !Equal(other);
    }

    bool Less(this_type const &right) const {
        auto const &left = *this;
        bool res = left.m_flag_.type < right.m_flag_.type;

        if (res) {
            switch (left.m_flag_.type & kTypeMask) {
                case kNull:
                    res = false;
                    break;
                case kObject:
                    res = (right.m_flag_.type == kObject) ? false : m_object_.ptr->Less(*right.m_object_.ptr);
                    break;
                case kArray:
                    res = (right.m_flag_.type == kObject) ? false : m_array_.ptr->Less(*right.m_array_.ptr);
                    break;
                case kChar: {
                    res = m_string_.Less(right.m_string_);
                } break;
                case kNumber:
                    res = m_number_.Less(right.m_number_);
                    break;
                default:
                    break;
            }
        }
        return res;
    }
    template <typename U>
    bool operator<(U const &right) const {
        return Less(value_type(right));
    }

    SpDMElement &operator=(value_type value) {
        Set(std::move(value));
        return *this;
    }

    auto flag() const { return m_flag_; }
    auto &data() { return m_data_; }
    auto const &data() const { return m_data_; }

    void reset() { value_type().swap(*this); };
    bool isNull() const { return m_flag_.type == kNull; }
    bool isBoolean() const { return (m_flag_.type & kTypeMask) == kBool; }
    bool isNumber() const { return (m_flag_.type & kTypeMask) == kNumber; }
    bool isString() const { return (m_flag_.type & kTypeMask) == kChar; }
    bool isObject() const { return (m_flag_.type & kTypeMask) == kObject; }
    bool isArray() const { return (m_flag_.type & kTypeMask) == kArray; }
    bool isTensor() const { return m_flag_.rank > 0 && ((m_flag_.type & kTypeMask) != kChar); }
    bool isLight() const { return isBoolean() || isNumber() || isString(); }
    bool empty() const { return isNull(); }
    //    bool isSoftLink() const { return ((m_flag_.type & kTypeMask) == kChar) && ((m_flag_.type & kIsLink) != 0);
    //    }

    explicit operator int() const { return as<int>(); }
    explicit operator unsigned int() const { return as<unsigned int>(); }
    explicit operator int64_t() const { return as<int64_t>(); }
    explicit operator uint64_t() const { return as<uint64_t>(); }
    explicit operator double() const { return as<double>(); }
    explicit operator bool() const { return m_flag_.type == kTrue; }
    explicit operator std::string() const { return as<std::string>(); }

   private:
    HAS_MEMBER_FUNCTION(Get)
    template <typename V, typename U>
    static auto get_helper_(V &v, U &u) -> std::enable_if_t<has_member_function_Get<V, U &>::value, int> {
        return v.Get(u);
    }
    template <typename V, typename U>
    static auto get_helper_(V &, U &) -> std::enable_if_t<!has_member_function_Get<V, U &>::value, int> {
        return Status::NotModified();
    }

   public:
    template <typename U>
    int Get(U &res) const {
        auto success = Status::NotModified();
        switch (m_flag_.type & kTypeMask) {
            case kObject:
                success = get_helper_(*traits::as_const(m_object_.ptr), res);
                break;
            case kArray:
                success = get_helper_(*traits::as_const(m_array_.ptr), res);
                break;
            case kBool:
            case kNumber:
                success = get_helper_(m_number_, res);
                break;
            case kChar:
                success = get_helper_(m_string_, res);
                break;
            default:
                break;
        }
        return success;
    }
    template <typename U>
    int Get(U &res) {
        auto success = Status::NotModified();
        switch (m_flag_.type & kTypeMask) {
            case kObject:
                success = get_helper_(*m_object_.ptr, res);
                break;
            case kArray:
                success = get_helper_(*m_object_.ptr, res);
                break;
            case kBool:
            case kNumber:
                success = get_helper_(m_number_, res);
                break;
            case kChar:
                success = get_helper_(m_string_, res);
                break;
            default:
                break;
        }
        return success;
    }
    template <typename U>
    auto as() const {
        U res;
        Get(res);
        return std::move(res);
    }
    template <typename U>
    auto as(U default_value) const {
        Get(default_value);
        return std::move(default_value);
    }

    /**
     *  convert element to Number
     * @return this
     */
    value_type *Number() {
        value_type *res = this;
        switch (m_flag_.type & kTypeMask) {
            case kChar:
                res = nullptr;
                break;
            case kArray: {
                auto p = res->m_array_.ptr->Number();
                if (!p.isNull()) { p.swap(*res); }
            } break;
            case kObject:
                if (m_object_.ptr->size() == 0) {
                    value_type(0).swap(*res);
                } else {
                    res = nullptr;
                }
                break;
            case kNull:
                value_type(0).swap(*res);
                break;
            case kBool:
            case kNumber:
            default:
                break;
        }
        return res;
    }
    /**
     *  convert element to String
     * @return this
     */
    value_type *String() {
        auto res = this;
        switch (m_flag_.type & kTypeMask) {
            case kNumber:
                value_type(as<std::string>()).swap(*res);
                break;
            case kArray:
            case kObject:
                throw(std::runtime_error("deleted function: convert object/array to string"));
            case kChar:
            default:
                break;
        }
        return res;
    }
    /**
     *  convert element to Array
     * @return
     */
    value_type *Array() {
        auto res = this;
        switch (m_flag_.type & kTypeMask) {
            case kArray:
                break;
            case kNull:
                this_type(kArray).swap(*res);
                break;
            default:
                this_type tmp(kArray);
                tmp.m_array_.ptr->Add(std::move(*res));
                tmp.swap(*this);
        }
        return res;
    }
    /**
     * convert element to Object
     * @return
     */
    value_type *Object() {
        auto res = this;
        switch (res->m_flag_.type & kTypeMask) {
            case kObject:
                break;
            case kNull:
                this_type(kObject).swap(*res);
                break;
            default:
                break;
                //                this_type tmp(kObject);
                //                tmp.m_object_.ptr->Insert("_")->Set(std::move(*res));
                //                tmp.swap(*res);
        }
        return res;
    }

    number_type &asNumber() { return Number()->m_number_; }
    string_type &asString() { return String()->m_string_; }
    array_type &asArray() { return *Array()->m_array_.ptr; }
    object_type &asObject() { return *Object()->m_object_.ptr; }

    number_type const &asNumber() const {
        assert(isNumber());
        return m_number_;
    }
    string_type const &asString() const {
        assert(isString());
        return m_string_;
    }

    array_type const &asArray() const {
        assert(isArray());
        return *m_array_.ptr;
    }
    object_type const &asObject() const {
        assert(isObject());
        return *m_object_.ptr;
    }

    object_type const *getObject() const { return (isObject()) ? m_object_.ptr : nullptr; }
    object_type *getObject() { return (isObject()) ? m_object_.ptr : nullptr; }
    array_type const *getArray() const { return (isArray()) ? m_array_.ptr : nullptr; }
    array_type *getArray() { return (isArray()) ? m_array_.ptr : nullptr; }

    template <typename U>
    this_type &operator=(U const &rhs) {
        Set(rhs);
        return *this;
    }

    //    auto Set(value_type v) {
    //        auto status = Status::NotModified();
    //        switch (m_flag_.type & kTypeMask) {
    //            case kNumber:
    //                break;
    //            case kObject:
    //                if (v.isObject()) {
    //                    status = m_object_.ptr->Set(std::move(*v.m_object_.ptr));
    //                    break;
    //                }
    //            case kNull:
    //            default:
    //                v.swap(*this);
    //                status = Status::OK();
    //                break;
    //        }
    //        return Status::OK();
    //    }
    //    template <typename... Args>
    //    auto Set(object_type const &obj) {
    //        if ((m_flag_.type & kTypeMask) == kObject) {
    //            return m_object_.ptr->Set(obj);
    //        } else {
    //            this_type(obj->Copy()).swap(*this);
    //            return Status::OK();
    //        }
    //    };
   private:
    static auto Set_(number_type &v, value_type &&u) { return v.Set(std::move(u.Number()->m_number_)); }
    static auto Set_(string_type &v, value_type &&u) { return v.Set(std::move(u.String()->m_string_)); }
    static auto Set_(object_type &v, value_type &&u) { return v.Set(std::move(u)); }
    static auto Set_(array_type &v, value_type &&u) { return v.Set(std::move(u)); }
    template <typename V, typename... Args>
    static auto Set_(V &v, Args &&... args)
        -> std::enable_if_t<detail::has_member_function_Set<V, std::remove_cv_t<Args>...>::value, int> {
        return v.Set(std::forward<Args>(args)...);
    }
    template <typename V, typename... Args>
    auto Set_(V &v, Args &&... args)
        -> std::enable_if_t<!detail::has_member_function_Set<V, std::remove_cv_t<Args>...>::value, int> {
        return Status::NotModified();
    }

   public:
    auto Set(value_type v) {
        auto status = Status::NotModified();
        switch (m_flag_.type & (kTypeMask | kIsReference)) {
            case kNumber | kIsReference:
            case kBool | kIsReference:
                status = Set_(m_number_, std::move(v));
                break;
            case kChar | kIsReference:
                status = Set_(m_string_, std::move(v));
                break;
            case kObject | kIsReference:
                status = Set_(*m_object_.ptr, std::move(v));
                break;
            case kArray | kIsReference:
                status = Set_(*m_array_.ptr, std::move(v));
                break;
            case kNull:
            default:
                v.swap(*this);
                status = Status::OK();
                break;
        }
        return status;
    }
    template <typename... Args>
    auto Set(Args &&... args) {
        auto status = Status::NotModified();
        switch (m_flag_.type & (kTypeMask | kIsReference)) {
            case kNumber | kIsReference:
            case kBool | kIsReference:
                status = Set_(m_number_, std::forward<Args>(args)...);
                break;
            case kChar | kIsReference:
                status = Set_(m_string_, std::forward<Args>(args)...);
                break;
            case kObject:
            case kObject | kIsReference:
                status = Set_(*m_object_.ptr, std::forward<Args>(args)...);
                break;
            case kArray:
            case kArray | kIsReference:
                status = Set_(*m_array_.ptr, std::forward<Args>(args)...);
                break;
            case kNull:
            default:
                this_type(std::forward<Args>(args)...).swap(*this);
                status = Status::OK();
                break;
        }

        return status;
    };

    template <typename... Args>
    value_type *Add(Args &&... args) {
        auto p = Array();
        if (p) {
            p = p->m_array_.ptr->Add();
            p->Set(std::forward<Args>(args)...);
        }
        return p;
    }

    template <typename... Args>
    auto Merge(Args &&... args) {
        auto success = Status::NotModified();
        switch (m_flag_.type & kTypeMask) {
            case kArray:
                success = m_array_.ptr->Merge(std::forward<Args>(args)...);
                break;
            case kObject:
                success = m_object_.ptr->Merge(std::forward<Args>(args)...);
                break;
            default:
                if (auto p = Add()) { success = p->Set(std::forward<Args>(args)...); }
                break;
        }
        return success;
    }

    // path dependent
    value_type *Insert(char_type const *str, size_type len) { return Insert(key_type(str, len)); }
    template <typename... Args>
    value_type *Insert(key_type const &k, Args &&... args) {
        value_type *res = this;
        switch (m_flag_.type & kTypeMask) {
            case kNull:
            case kBool:
            case kNumber:
            case kChar:
                res = Object();
            case kObject:
                res = m_object_.ptr->Insert(k, std::forward<Args>(args)...);
                break;
            case kArray:
                res = m_array_.ptr->Insert(k, std::forward<Args>(args)...);
                break;
            default:
                break;
        }
        return res;
    }
    template <typename... Args>
    value_type *InsertOrAssign(key_type const &k, Args &&... args) {
        value_type *res = this;
        switch (m_flag_.type & kTypeMask) {
            case kNull:
            case kBool:
            case kNumber:
            case kChar:
                res = Object();
            case kObject:
                res = m_object_.ptr->InsertOrAssign(k, std::forward<Args>(args)...);
                break;
            case kArray:
                res = m_array_.ptr->InsertOrAssign(k, std::forward<Args>(args)...);
                break;
            default:
                break;
        }

        return res;
    }
    auto Find(key_type const &k) {
        value_type *res = nullptr;
        switch (m_flag_.type & kTypeMask) {
            case kArray:
                res = m_array_.ptr->Find(k);
                break;
            case kObject:
                res = m_object_.ptr->Find(k);
                break;
            default:
                break;
        }
        return res;
    }
    auto Find(key_type const &k) const {
        value_type const *res = nullptr;
        switch (m_flag_.type & kTypeMask) {
            case kArray:
                res = traits::as_const(m_array_.ptr)->Find(k);
                break;
            case kObject:
                res = traits::as_const(m_object_.ptr)->Find(k);
                break;
            default:
                break;
        }
        return res;
    }
    auto Delete(key_type const &k) {
        auto success = Status::NotModified();
        switch (m_flag_.type & kTypeMask) {
            case kArray:
                success = m_array_.ptr->Delete(k);
                break;
            case kObject:
                success = m_object_.ptr->Delete(k);
                break;
            default:
                break;
        }
        return success;
    }

    template <typename K>
    size_type size(K const &k) const {
        if (auto p = Find(k)) {
            return p->size();
        } else {
            return 0;
        }
    }
    template <typename K, typename... Args>
    auto Insert(K const &k, Args &&... args) -> std::enable_if_t<!std::is_same<K, key_type>::value, value_type *> {
        return path_type(k).Insert(*this, std::forward<Args>(args)...);
    }
    template <typename K, typename... Args>
    auto InsertOrAssign(K const &k, Args &&... args)
        -> std::enable_if_t<!std::is_same<K, key_type>::value, value_type *> {
        return path_type(k).InsertOrAssign(*this, std::forward<Args>(args)...);
    }

    template <typename K>
    auto Find(K const &k) const {
        return path_type(k).Find(*this);
    }
    template <typename K>
    auto Find(K const &k) {
        return path_type(k).Find(*this);
    }
    template <typename K>
    decltype(auto) at(K const &k) const {
        if (auto p = Find(k)) {
            return *p;
        } else {
            throw(std::out_of_range(traits::to_string(k)));
        }
    }
    template <typename K>
    decltype(auto) at(K const &k) {
        if (auto p = Find(k)) {
            return *p;
        } else {
            throw(std::out_of_range(traits::to_string(k)));
        }
    }
    template <typename K>
    auto get(K const &k) {
        return SpDMReference<this_type>{this, k};
    }
    //    template <typename U, typename K>
    //    U as(K const &k) const {
    //        return at(k).template as<U>();
    //    }
    //    template <typename U, typename K>
    //    U as(K const &k, U default_value) const {
    //        return at(k)->template as<U>(default_value);
    //    }
    template <typename K>
    decltype(auto) operator[](K const &k) {
        return get(k);
    }
    template <typename K>
    decltype(auto) operator[](K const &k) const {
        return at(k);
    }
    template <typename K>
    int Delete(K const &k) {
        return path_type(k).Delete(*this);
    }

    // visitor

    int Accept(visitor_type const &entry) const {
        int count = 0;

        switch (m_flag_.type & kTypeMask) {
            case kBool:
            case kNumber:
                count += m_number_.Accept(entry);
                break;
            case kChar:
                count += m_string_.Accept(entry);
                break;
            case kObject:
                count += traits::as_const(m_object_.ptr)->Accept(entry);
                break;
            case kArray:
                count += traits::as_const(m_array_.ptr)->Accept(entry);
                break;
            case kNull:
            default:
                count += entry.Null();
                break;
        }

        return count;
    }
    template <typename VISITOR>
    bool Accept(VISITOR &&visitor,
                std::enable_if_t<!std::is_base_of<visitor_type, traits::remove_cvref_t<VISITOR>>::value> *sfinae =
                    nullptr) const {
        return Accept(visitor_type(std::forward<VISITOR>(visitor)));
    }
    bool Accept(visitor_interface_type &&visitor) const { return Accept(visitor_type(std::move(visitor))); }

    template <typename U>
    bool Serialize(U &&serializer) const {
        return Accept(std::forward<U>(serializer));
    }

    // place holder
    this_type operator,(this_type const &other) const {
        this_type res(*this);
        res.Merge(other);
        return std::move(res);
    };
    this_type operator,(this_type &&other) {
        this_type res(*this);
        res.Merge(std::move(other));
        return std::move(res);
    };
    struct place_holder {
        key_type m_key_;
        place_holder(place_holder const &other) : m_key_(other.m_key_) {}
        place_holder(place_holder &&other) noexcept : m_key_(std::move(other.m_key_)) {}
        template <typename... Args>
        explicit place_holder(Args &&... args) : m_key_(std::forward<Args>(args)...) {}

        ~place_holder() = default;

        template <typename U>
        value_type operator=(U &&u) {
            value_type res;
            res.InsertOrAssign(m_key_, value_type(std::forward<U>(u)));
            return std::move(res);
        };

        template <typename U>
        value_type operator=(std::initializer_list<U> const &u) {
            value_type res;
            res.InsertOrAssign(m_key_, u);
            return std::move(res);
        };
        template <typename U>
        value_type operator=(std::initializer_list<std::initializer_list<U>> const &u) {
            value_type res;
            res.InsertOrAssign(m_key_, u);
            return std::move(res);
        };
        template <typename U>
        value_type operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const &u) {
            value_type res;
            res.InsertOrAssign(m_key_, value_type(u));
            return std::move(res);
        };
    };

};  // struct SpDMElement

inline SpDMElement<>::place_holder operator"" _(const char *c, std::size_t n) {
    return SpDMElement<>::place_holder(c, n);
}
using SpDOM = SpDMObject<>;
using spObject = SpDMObject<>;
using spArray = SpDMArray<>;
using DataEntry = SpDMElement<>;
using SpDataEntry = SpDMElement<>;
using spString = SpDMString<>;
}  // namespace simpla

#endif  // SPDM_SPDM_H
