#include "../db/DataBlock.h"
#include "../db/Node.h"
#include "../db/NodePlugin.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <variant>
extern "C"
{
#include <hdf5.h>
#include <hdf5_hl.h>
}

namespace sp::db
{

#define H5_ERROR(_FUN_)                                    \
    if ((_FUN_) < 0)                                       \
    {                                                      \
        H5Eprint(H5E_DEFAULT, stderr);                     \
        RUNTIME_ERROR << "HDF5 Error:" << __STRING(_FUN_); \
    }

template <typename T>
struct h5type_traits
{
    static hid_t type()
    {
        NOT_IMPLEMENTED;
        return H5Tcopy(H5T_OPAQUE);
    }
    static hid_t space() { return H5Screate(H5S_SCALAR); }
    static const void* data(const T& v) { return &v; }
};

#define DEC_TYPE(_T_, _H5_T_) \
    template <>               \
    hid_t h5type_traits<_T_>::type() { return H5Tcopy(_H5_T_); }

DEC_TYPE(bool, H5T_NATIVE_HBOOL)
DEC_TYPE(float, H5T_NATIVE_FLOAT)
DEC_TYPE(double, H5T_NATIVE_DOUBLE)
DEC_TYPE(int, H5T_NATIVE_INT)
DEC_TYPE(long, H5T_NATIVE_LONG)
DEC_TYPE(unsigned int, H5T_NATIVE_UINT)
DEC_TYPE(unsigned long, H5T_NATIVE_ULONG)
#undef DEC_TYPE

template <typename T, std::size_t N>
struct h5type_traits<std::array<T, N>>
{

    static hid_t type()
    {
        return h5type_traits<T>::type();
    }

    static hid_t space()
    {
        hsize_t d = N;
        return H5Screate_simple(1, &d, nullptr);
    }

    static void* data(const std::array<T, N>& d) { return d.data(); }
};

struct hdf5_node
{
    std::shared_ptr<hid_t> fid;
    std::shared_ptr<hid_t> oid;
};

typedef NodePlugin<hdf5_node> NodePluginHDF5;

hid_t h5_open_group(hid_t base, const Path& path, bool create_if_not_exists = true)
{
    hid_t last = base;
    auto it = path.begin();
    auto ie = path.end();

    Path prefix;

    Path::Segment item;

    while (it != ie)
    {
        Path::Segment(*it).swap(item);

        prefix.append(item);

        ++it;

        hid_t next = -1;

        std::visit(
            traits::overloaded{
                [&](const std::variant_alternative_t<Path::tags::Key, Path::Segment>& key) {
                    if (H5Lexists(last, key.c_str(), H5P_DEFAULT) > 0)
                    {
                        H5_ERROR(next = H5Oopen(last, key.c_str(), H5P_DEFAULT));
                    }
                    else if (create_if_not_exists)
                    {
                        H5_ERROR(next = H5Gcreate(last, key.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                    }
                    else
                    {
                        next = -1;
                    }
                },
                [&](const std::variant_alternative_t<Path::tags::Index, Path::Segment>& i) {
                    int idx = i;
                    H5O_info_t oinfo;

                    H5_ERROR(H5Oget_info(last, &oinfo));

                    if (oinfo.type == H5O_TYPE_GROUP)
                    {
                        H5G_info_t ginfo;

                        H5_ERROR(H5Gget_info(last, &ginfo));

                        if (idx < 0 && ginfo.nlinks > 0)
                        {
                            idx = (idx + ginfo.nlinks) % ginfo.nlinks;
                        }

                        if (idx < ginfo.nlinks)
                        {
                            H5_ERROR(next = H5Oopen_by_idx(last, ".", H5_INDEX_NAME, H5_ITER_INC, static_cast<hsize_t>(idx), H5P_DEFAULT));
                        }
                        else
                        {
                            THROW_EXCEPTION_OUT_OF_RANGE(idx, " >= ", ginfo.nlinks);
                        }
                    }
                    else if (oinfo.type == H5O_TYPE_DATASET)
                    {
                        NOT_IMPLEMENTED;
                        // TODO: slice
                    }
                    else
                    {
                        NOT_IMPLEMENTED;
                    }
                },
                [&](const std::variant_alternative_t<Path::tags::Slice, Path::Segment>& slice) {
                    NOT_IMPLEMENTED;
                },
                [&](auto&&) { NOT_IMPLEMENTED; }},
            item);

        if (last != base) H5Oclose(last);

        if (next < 0)
        {
            THROW_EXCEPTION_OUT_OF_RANGE(prefix);
        }

        last = next;
    }

    return last;
}

template <>
void NodePluginHDF5::load(const Node& opt)
{

    m_container_.fid = std::shared_ptr<hid_t>(new hid_t(-1), [&](hid_t* p) {
        if (p != nullptr && *p >= 0) H5_ERROR(H5Fclose(*p));
        delete p;
    });
    m_container_.oid = std::shared_ptr<hid_t>(new hid_t(-1), [&](hid_t* p) {
        if (p != nullptr && *p >= 0) H5_ERROR(H5Gclose(*p));
        delete p;
    });

    std::string file_path;
    std::string group_path = "/";
    std::string mode = "create";
    opt.visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& path) { file_path = path; },
            [&](const std::variant_alternative_t<Node::tags::Path, Node::value_type>& path) { file_path = path.str(); },
            [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                file_path = object_p->find_child("file").get_value<std::string>("unnamed.h5");
                group_path = object_p->find_child("path").get_value<std::string>("/");
                mode = object_p->find_child("mode").get_value<std::string>("create");
            },
            [&](auto&& ele) { NOT_IMPLEMENTED; } //
        });

    if (mode == "create")
    {
        H5_ERROR(*m_container_.fid = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
        H5_ERROR(*m_container_.oid = h5_open_group(*m_container_.fid, Path::parse(group_path), true));

        VERBOSE << "Create HDF5 file: " << file_path << " : " << group_path;
    }
    else
    {
        H5_ERROR(*m_container_.fid = H5Fopen(file_path.c_str(), H5P_DEFAULT, H5P_DEFAULT));
        H5_ERROR(*m_container_.oid = h5_open_group(*m_container_.fid, Path::parse(group_path), false));
        VERBOSE << "Open HDF5 file: " << file_path << " : " << group_path;
    }
}

template <>
void NodePluginHDF5::save(const Node& node) const
{
}

template <>
Cursor<const Node>
NodePluginHDF5::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Node>{};
}

template <>
Cursor<Node>
NodePluginHDF5::children()
{
    NOT_IMPLEMENTED;
    return Cursor<Node>{};
}

template <>
void NodePluginHDF5::for_each(std::function<void(const Node&, const Node&)> const& visitor) const
{
}

Node h5_update_attribute(hid_t gid, const std::string& name, const Node& data)
{

    Node res;

    // if (H5Lexists(gid, name.c_str(), H5P_DEFAULT) > 0)
    // {
    //     RUNTIME_ERROR << "Can not rewrite exist dataset/group! [" << name << "]";
    // }
    // else
    if (H5Aexists(gid, name.c_str()) > 0)
    {
        H5_ERROR(H5Adelete(gid, name.c_str()));
    }

    data.visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& p_object) {
                VERBOSE << name;
                NOT_IMPLEMENTED;
            },
            [&](const std::variant_alternative_t<Node::tags::Array, Node::value_type>& p_array) {
                std::vector<char const*> s_array;
                p_array->for_each([&](const Node&, const Node& v) {
                    s_array.push_back(v.as<Node::tags::String>().c_str());
                });

                hsize_t s = s_array.size();
                auto m_space = H5Screate_simple(1, &s, nullptr);
                auto m_type = H5Tcopy(H5T_C_S1);
                H5_ERROR(H5Tset_size(m_type, H5T_VARIABLE));
                auto aid = H5Acreate(gid, name.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
                H5_ERROR(H5Awrite(aid, m_type, &s_array[0]));
                H5_ERROR(H5Tclose(m_type));
                H5_ERROR(H5Sclose(m_space));
                H5_ERROR(H5Aclose(aid));
            },
            [&](const std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) {
                NOT_IMPLEMENTED;
            },
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& s_str) {
                auto m_type = H5Tcopy(H5T_C_S1);
                H5_ERROR(H5Tset_size(m_type, s_str.size()));
                H5_ERROR(H5Tset_strpad(m_type, H5T_STR_NULLTERM));
                auto m_space = H5Screate(H5S_SCALAR);
                auto aid = H5Acreate(gid, name.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
                H5_ERROR(H5Awrite(aid, m_type, s_str.c_str()));
                H5_ERROR(H5Tclose(m_type));
                H5_ERROR(H5Sclose(m_space));
                H5_ERROR(H5Aclose(aid));
            },
            [&](const std::variant_alternative_t<Node::tags::Complex, Node::value_type>& complex) {
                NOT_IMPLEMENTED;
            },
            [&](auto&& v) {
                typedef std::remove_const_t<std::remove_reference_t<decltype(v)>> T;
                hid_t d_type = h5type_traits<T>::type();
                hid_t d_space = h5type_traits<T>::space();
                hid_t aid = 0;
                H5_ERROR(aid = H5Acreate(gid, name.c_str(), d_type, d_space, H5P_DEFAULT, H5P_DEFAULT));
                H5_ERROR(H5Awrite(aid, d_type, h5type_traits<T>::data(v)));
                H5_ERROR(H5Aclose(aid));
                H5_ERROR(H5Sclose(d_space));
                H5_ERROR(H5Tclose(d_type));
            }

        });

    return std::move(res);
}

Node h5_update(hid_t gid, const std::string& key, const NodeObject& obj)
{
    Node res;

    hid_t oid = gid;

    NOT_IMPLEMENTED;

    if (oid != gid)
    {
        H5Oclose(oid);
    }
    return res;
}
Node h5_update(hid_t gid, const std::string& key, const NodeArray& data) { NOT_IMPLEMENTED; }
Node h5_update(hid_t gid, const std::string& key, const DataBlock& data) { NOT_IMPLEMENTED; }

Node h5_update(hid_t gid, int idx, const Node& data)
{
    Node res;
    H5G_info_t ginfo;
    H5_ERROR(H5Gget_info(gid, &ginfo));
    if (idx < ginfo.nlinks && idx >= 0)
    {
        hid_t oid = -1;
        H5_ERROR(oid = H5Oopen_by_idx(gid, ".", H5_INDEX_NAME, H5_ITER_INC, (hsize_t)idx, H5P_DEFAULT));
        h5_update(oid, "", data).swap(res);
        H5_ERROR(H5Oclose(oid));
    }
    else
    {
        THROW_EXCEPTION_OUT_OF_RANGE(idx, ">= ", ginfo.nlinks);
    }
    return std::move(res);
}
Node h5_update(hid_t gid, const std::string& key, const Node& data)
{

    Node res;

    size_t count = 0;

    if (H5Lexists(gid, key.c_str(), H5P_DEFAULT) > 0)
    {
        RUNTIME_ERROR << "Can not rewrite exist dataset/group! [" << key << "]";
    }
    else if (H5Aexists(gid, key.c_str()) > 0)
    {
        H5_ERROR(H5Adelete(gid, key.c_str()));
    }

    data.visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& p_object) {
                NOT_IMPLEMENTED;
            },
            [&](const std::variant_alternative_t<Node::tags::Array, Node::value_type>& p_array) {
                std::vector<char const*> s_array;
                p_array->for_each([&](const Node&, const Node& v) {
                    s_array.push_back(v.as<Node::tags::String>().c_str());
                });

                hsize_t s = s_array.size();
                auto m_space = H5Screate_simple(1, &s, nullptr);
                auto m_type = H5Tcopy(H5T_C_S1);
                H5_ERROR(H5Tset_size(m_type, H5T_VARIABLE));
                auto aid = H5Acreate(gid, key.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
                H5_ERROR(H5Awrite(aid, m_type, &s_array[0]));
                H5_ERROR(H5Tclose(m_type));
                H5_ERROR(H5Sclose(m_space));
                H5_ERROR(H5Aclose(aid));
                count = s;
            },
            [&](const std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) {
                NOT_IMPLEMENTED;
            },
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& s_str) {
                auto m_type = H5Tcopy(H5T_C_S1);
                H5_ERROR(H5Tset_size(m_type, s_str.size()));
                H5_ERROR(H5Tset_strpad(m_type, H5T_STR_NULLTERM));
                auto m_space = H5Screate(H5S_SCALAR);
                auto aid = H5Acreate(gid, key.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
                H5_ERROR(H5Awrite(aid, m_type, s_str.c_str()));
                H5_ERROR(H5Tclose(m_type));
                H5_ERROR(H5Sclose(m_space));
                H5_ERROR(H5Aclose(aid));
                ++count;
            },
            [&](const std::variant_alternative_t<Node::tags::Complex, Node::value_type>& complex) {
                NOT_IMPLEMENTED;
            },
            [&](auto&& v) {
                typedef std::remove_const_t<std::remove_reference_t<decltype(v)>> T;
                hid_t d_type = h5type_traits<T>::type();
                hid_t d_space = h5type_traits<T>::space();
                hid_t aid = 0;
                H5_ERROR(aid = H5Acreate(gid, key.c_str(), d_type, d_space, H5P_DEFAULT, H5P_DEFAULT));
                H5_ERROR(H5Awrite(aid, d_type, h5type_traits<T>::data(v)));
                H5_ERROR(H5Aclose(aid));
                H5_ERROR(H5Sclose(d_space));
                H5_ERROR(H5Tclose(d_type));
                ++count;
            }

        });

    // data.visit(
    //     traits::overloaded{
    //         [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& p_object) {
    //             oid = h5_open_group(base, path.prefix(), true);
    //             h5_update(oid, path.last(), *p_object);
    //         },
    //         [&](const std::variant_alternative_t<Node::tags::Array, Node::value_type>& p_array) {
    //             if (p_array->is_simple())
    //             {
    //                 oid = h5_open_group(base, path.prefix(), true);
    //                 h5_update_attribute(oid, std::get<std::string>(path.last()), data).swap(res);
    //             }
    //             else
    //             {
    //                 oid = h5_open_group(base, path, true);
    //                 h5_update(oid, nullptr, *p_array);
    //             }
    //         },
    //         [&](const std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) {
    //             oid = h5_open_group(base, path.prefix(), true);
    //             h5_update(oid, path.last(), blk);
    //         },
    //         [&](const std::variant_alternative_t<Node::tags::Path, Node::value_type>& p) { NOT_IMPLEMENTED; },
    //         [&](auto&& v) {
    //             oid = h5_open_group(base, path.prefix(), true);
    //             h5_update_attribute(oid, std::get<std::string>(path.last()), data).swap(res);
    //         },
    //     });

    return res;
}

template <>
Node NodePluginHDF5::update(const Path& path, int op, const Node& data)
{

    hid_t base = path.is_absolute() ? *m_container_.fid : *m_container_.oid;

    hid_t gid = h5_open_group(base, path.prefix(), true);

    Node res;

    std::visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Path::tags::Key, Path::Segment>& key) {
                h5_update(gid, key, data);
            },
            [&](const std::variant_alternative_t<Path::tags::Index, Path::Segment>& idx) {
                h5_update(gid, idx, data);
            },
            [&](const std::variant_alternative_t<Path::tags::Slice, Path::Segment>& slice) {
                auto& array = res.as_array();
                slice.for_each([&](int idx) { h5_update(gid, idx, data).swap(array.push_back()); });
            },
            [&](auto&& v) { NOT_IMPLEMENTED; }},
        path.last());

    if (gid != base) H5Oclose(gid);

    return std::move(res);
}

Node h5_fetch_object(hid_t oid, const Node& projection)
{
    Node res{};

    NOT_IMPLEMENTED;

    return res;
}

Node h5_fetch_attribute(hid_t oid, const Node& projection)
{
    Node res;

    hid_t d_type = H5Aget_type(oid);
    hid_t d_space = H5Aget_space(oid);

    H5T_class_t type_class = H5Tget_class(d_type);

    if (type_class == H5T_INTEGER)
    {
        if (H5Tequal(d_type, H5T_NATIVE_HBOOL) > 0)
        {
            bool v;
            H5_ERROR(H5Aread(oid, H5T_NATIVE_HBOOL, &v));
            res.set_value<Node::tags::Boolean>(v);
        }
        else
        {
            int buff;
            H5_ERROR(H5Aread(oid, H5T_NATIVE_INT, &buff));
            res.set_value<Node::tags::Integer>(buff);
        }
    }
    else if (type_class == H5T_FLOAT)
    {
        double buff;
        H5_ERROR(H5Aread(oid, H5T_NATIVE_DOUBLE, &buff));
        res.set_value<Node::tags::Double>(buff);
    }
    else if (type_class == H5T_STRING)
    {
        size_t sdims = H5Tget_size(d_type);
        char buffer[sdims + 1];
        auto m_type = H5Tcopy(H5T_C_S1);
        H5_ERROR(H5Tset_size(m_type, sdims));
        H5_ERROR(H5Aread(oid, m_type, buffer));
        H5_ERROR(H5Tclose(m_type));
        res.set_value<Node::tags::String>(std::string(buffer));
    }
    else if (type_class == H5T_ARRAY)
    {
        NOT_IMPLEMENTED;
    }
    else if (type_class == H5T_TIME)
    {
        NOT_IMPLEMENTED;
    }
    else if (type_class == H5T_BITFIELD)
    {
        NOT_IMPLEMENTED;
    }
    else if (type_class == H5T_REFERENCE)
    {
        NOT_IMPLEMENTED;
    }
    else if (type_class == H5T_ENUM)
    {
        NOT_IMPLEMENTED;
    }
    else if (type_class == H5T_VLEN)
    {
        NOT_IMPLEMENTED;
    }
    else if (type_class == H5T_OPAQUE)
    {
        NOT_IMPLEMENTED;
    }
    else if (type_class == H5T_COMPOUND)
    {
        NOT_IMPLEMENTED;
    }
    else if (type_class == H5T_NO_CLASS)
    {
        NOT_IMPLEMENTED;
    }

    H5Tclose(d_type);
    H5Sclose(d_space);

    return res;
}

template <typename U>
Node h5_fetch(hid_t gid, const U& idx, const Node& ops)
{
    NOT_IMPLEMENTED;
    return Node{};
}

Node h5_fetch(hid_t gid, int idx, const Node& ops)
{
    Node res;
    H5G_info_t ginfo;
    H5_ERROR(H5Gget_info(gid, &ginfo));
    if (idx < ginfo.nlinks && idx >= 0)
    {
        hid_t oid = -1;
        H5_ERROR(oid = H5Oopen_by_idx(gid, ".", H5_INDEX_NAME, H5_ITER_INC, (hsize_t)idx, H5P_DEFAULT));
        h5_fetch_object(oid, ops).swap(res);
        H5_ERROR(H5Oclose(oid));
    }
    else
    {
        THROW_EXCEPTION_OUT_OF_RANGE(idx, ">= ", ginfo.nlinks);
    }
    return std::move(res);
}

Node h5_fetch(hid_t gid, const std::string& key, const Node& ops)
{
    Node res;
    if (H5Lexists(gid, key.c_str(), H5P_DEFAULT) > 0 && H5Oexists_by_name(gid, key.c_str(), H5P_DEFAULT) > 0)
    {
        hid_t oid = -1;
        H5_ERROR(oid = H5Oopen(gid, key.c_str(), H5P_DEFAULT));
        h5_fetch_object(oid, ops).swap(res);
        H5_ERROR(H5Oclose(oid));
    }
    else if (H5Aexists(gid, key.c_str()) > 0)
    {
        hid_t aid = -1;
        H5_ERROR(aid = H5Aopen(gid, key.c_str(), H5P_DEFAULT));
        h5_fetch_attribute(aid, ops).swap(res);
        H5_ERROR(H5Aclose(aid));
    }
    else
    {
        Node(ops).swap(res);
    }
    return std::move(res);
}

template <>
const Node NodePluginHDF5::fetch(const Path& path, int op, const Node& ops) const
{
    hid_t base = path.is_absolute() ? *m_container_.fid : *m_container_.oid;

    Node res;

    hid_t gid = h5_open_group(base, path.prefix(), false);

    std::visit(
        traits::overloaded{
            // [&](const std::variant_alternative_t<Path::tags::Key, Path::Segment>& key) { h5_fetch(gid, key, ops).swap(res); },
            // [&](const std::variant_alternative_t<Path::tags::Index, Path::Segment>& idx) { h5_fetch(gid, idx, ops).swap(res); },
            [&](const std::variant_alternative_t<Path::tags::Slice, Path::Segment>& slice) {
                auto& array = res.as_array();
                slice.for_each([&](int idx) { h5_fetch(gid, idx, ops).swap(array.push_back()); });
            },
            [&](auto&& v) {
                h5_fetch(gid, v, ops).swap(res);
            }} // namespace sp::db
        ,
        path.last());

    if (gid != base) H5Oclose(gid);

    return res;
}

SPDB_ENTRY_REGISTER(hdf5, hdf5_node);
SPDB_ENTRY_ASSOCIATE(hdf5, hdf5_node, "^(.*)\\.(hdf5|h5)$");

} // namespace sp::db