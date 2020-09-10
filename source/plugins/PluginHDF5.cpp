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
namespace _detail
{
#define H5_ERROR(_FUN_)                                    \
    if ((_FUN_) < 0)                                       \
    {                                                      \
        H5Eprint(H5E_DEFAULT, stderr);                     \
        RUNTIME_ERROR << "HDF5 Error:" << __STRING(_FUN_); \
    }

hid_t H5NumberType(std::type_info const& t_info);

template <template <typename> class TFun, typename... Args>
void H5TypeDispatch(hid_t d_type, Args&&... args)
{
    H5T_class_t type_class = H5Tget_class(d_type);

    if ((type_class == H5T_INTEGER || type_class == H5T_FLOAT))
    {
        if (H5Tequal(d_type, H5T_NATIVE_CHAR) > 0)
        {
            TFun<char>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_SHORT) > 0)
        {
            TFun<short>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_INT) > 0)
        {
            TFun<int>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_LONG) > 0)
        {
            TFun<double>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_LLONG) > 0)
        {
            TFun<long long>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_UCHAR) > 0)
        {
            TFun<unsigned char>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_USHORT) > 0)
        {
            TFun<unsigned short>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_UINT) > 0)
        {
            TFun<unsigned int>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_ULONG) > 0)
        {
            TFun<unsigned long>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_ULLONG) > 0)
        {
            TFun<unsigned long long>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_FLOAT) > 0)
        {
            TFun<float>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_DOUBLE) > 0)
        {
            TFun<double>(std::forward<Args>(args)...);
        }
        else if (H5Tequal(d_type, H5T_NATIVE_LDOUBLE) > 0)
        {
            TFun<long double>(std::forward<Args>(args)...);
        }
    }
    else if (type_class == H5T_ARRAY)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_STRING)
    {
        TFun<std::string>(std::forward<Args>(args)...);
    }
    else if (type_class == H5T_TIME)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_BITFIELD)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_REFERENCE)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_ENUM)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_VLEN)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_NO_CLASS)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_OPAQUE)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_COMPOUND)
    {
        UNIMPLEMENTED;
    }
}

std::pair<hid_t, std::string> H5GroupTryOpen(hid_t root, std::string const& url, bool create_if_not_exist = true)
{
    hid_t gid = root;

    size_t begin = 0;

    while (true)
    {
        size_t pos = url.find('/', begin);

        if (pos == std::string::npos)
        {
            break;
        }
        else if (pos == begin)
        {
            ++begin;
            continue;
        }

        auto key = url.substr(begin, pos - begin);

        hid_t tid = 0;

        if (H5Lexists(gid, key.c_str(), H5P_DEFAULT) > 0)
        {
            H5O_info_t o_info;
            H5_ERROR(H5Oget_info_by_name(gid, key.c_str(), &o_info, H5P_DEFAULT));
            if (o_info.type == H5O_TYPE_GROUP)
            {
                tid = H5Gopen(gid, key.c_str(), H5P_DEFAULT);
            }
            else
            {
                RUNTIME_ERROR << key << " is  a  dataset!";
            }
        }
        else if (H5Aexists(gid, key.c_str()) > 0)
        {
            RUNTIME_ERROR << key << " is  an attribute!";
        }
        else
        {
            tid = H5Gcreate(gid, key.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }

        if (gid != root)
        {
            H5Gclose(gid);
        }
        gid = tid;
        begin = pos + 1;
    };
    return std::make_pair(gid, url.substr(begin));
}

hid_t HDF5CreateOrOpenGroup(hid_t grp, std::string const& key, bool create_if_not_exist)
{
    VERBOSE << key;

    hid_t res;
    if (H5Lexists(grp, key.c_str(), H5P_DEFAULT) > 0)
    {
        H5O_info_t o_info;
        H5_ERROR(H5Oget_info_by_name(grp, key.c_str(), &o_info, H5P_DEFAULT));
        if (o_info.type == H5O_TYPE_GROUP)
        {
            res = H5Gopen(grp, key.c_str(), H5P_DEFAULT);
        }
        else
        {
            RUNTIME_ERROR << key << " is  a  dataset!";
        }
    }
    else if (H5Aexists(grp, key.c_str()) > 0)
    {
        RUNTIME_ERROR << key << " is  an attribute!";
    }
    else
    {
        res = H5Gcreate(grp, key.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    return res;
}

hid_t GetHDF5DataType(std::type_info const& t_info);

template <typename U>
Node HDF5GetValueT(hid_t attr_id, hid_t d_type, hid_t d_space, bool is_attribute)
{
    Node res;
    if (is_attribute)
    {
        switch (H5Sget_simple_extent_type(d_space))
        {
        case H5S_SCALAR:
        {
            U v;
            H5_ERROR(H5Aread(attr_id, d_type, &v));
            res.set_value<U>(v);
        }
        break;
        case H5S_SIMPLE:
        {
            int ndims = H5Sget_simple_extent_ndims(d_space);
            hsize_t dims[ndims];
            H5_ERROR(H5Sget_simple_extent_dims(d_space, dims, nullptr));
            U* data = (new U[H5Sget_simple_extent_npoints(d_space)]);
            H5_ERROR(H5Aread(attr_id, d_type, data));
            NOT_IMPLEMENTED;
            // res.set_value<Node::tags::Block>(ndims, dims, data);
        }
        break;
        case H5S_NULL:
        default:
            break;
        }
    }
    else
    {
        NOT_IMPLEMENTED;
    }
    return res;
}

hid_t GetHDF5DataType(std::type_info const& t_info)
{
    hid_t v_type = H5T_NO_CLASS;

    if (t_info == typeid(char))
    {
        v_type = H5T_NATIVE_CHAR;
    }
    else if (t_info == typeid(int))
    {
        v_type = H5T_NATIVE_INT;
    }
    else if (t_info == typeid(long))
    {
        v_type = H5T_NATIVE_LONG;
    }
    else if (t_info == typeid(unsigned int))
    {
        v_type = H5T_NATIVE_UINT;
    }
    else if (t_info == typeid(unsigned long))
    {
        v_type = H5T_NATIVE_ULONG;
    }
    else if (t_info == typeid(float))
    {
        v_type = H5T_NATIVE_FLOAT;
    }
    else if (t_info == typeid(double))
    {
        v_type = H5T_NATIVE_DOUBLE;
    }
    else if (t_info == typeid(std::complex<double>))
    {
        H5_ERROR(v_type = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>)));
        H5_ERROR(H5Tinsert(v_type, "r", 0, H5T_NATIVE_DOUBLE));
        H5_ERROR(H5Tinsert(v_type, "i", sizeof(double), H5T_NATIVE_DOUBLE));
    }
    // TODO: array ptr type
    //   else if (d_type->isArray()) {
    //        auto const& t_array = d_type->cast_as<DataArray>();
    //        hsize_t dims[t_array.rank()];
    //        for (int i = 0; i < t_array.rank(); ++i) { dims[i] = t_array.dimensions()[i]; }
    //        hid_t res2 = res;
    //        H5_ERROR(res2 = H5Tarray_create(res, t_array.rank(), dims));
    //        if (H5Tcommitted(res) > 0) H5_ERROR(H5Tclose(res));
    //        res = res2;
    //    } else if (d_type->isObject()) {
    //        H5_ERROR(v_type = H5Tcreate(H5T_COMPOUND, d_type.size_in_byte()));
    //
    //        for (auto const& item : d_type.members()) {
    //            hid_t t_member = convert_data_type_sp_to_h5(std::get<0>(item), true);
    //
    //            H5_ERROR(H5Tinsert(res, std::get<1>(item).c_str(), std::get<2>(item), t_member));
    //            if (H5Tcommitted(t_member) > 0) H5_ERROR(H5Tclose(t_member));
    //        }
    //    }
    else
    {
        RUNTIME_ERROR << "Unknown m_data type:" << t_info.name();
    }
    //
    return (v_type);
}

hid_t H5NumberType(std::type_info const& t_info)
{
    hid_t v_type = H5T_NO_CLASS;
    if (t_info == typeid(char))
    {
        v_type = H5T_NATIVE_CHAR;
    }
    else if (t_info == typeid(int))
    {
        v_type = H5T_NATIVE_INT;
    }
    else if (t_info == typeid(long))
    {
        v_type = H5T_NATIVE_LONG;
    }
    else if (t_info == typeid(unsigned int))
    {
        v_type = H5T_NATIVE_UINT;
    }
    else if (t_info == typeid(unsigned long))
    {
        v_type = H5T_NATIVE_ULONG;
    }
    else if (t_info == typeid(float))
    {
        v_type = H5T_NATIVE_FLOAT;
    }
    else if (t_info == typeid(double))
    {
        v_type = H5T_NATIVE_DOUBLE;
    }
    else if (t_info == typeid(std::complex<double>))
    {
        H5_ERROR(v_type = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>)));
        H5_ERROR(H5Tinsert(v_type, "r", 0, H5T_NATIVE_DOUBLE));
        H5_ERROR(H5Tinsert(v_type, "i", sizeof(double), H5T_NATIVE_DOUBLE));
    }
    return v_type;
}

Node HDF5GetValue(hid_t obj_id, bool is_attribute)
{
    Node res = nullptr;
    hid_t d_type, d_space;
    if (is_attribute)
    {
        d_type = H5Aget_type(obj_id);
        d_space = H5Aget_space(obj_id);
    }
    else
    {
        d_type = H5Dget_type(obj_id);
        d_space = H5Dget_space(obj_id);
    }

    H5T_class_t type_class = H5Tget_class(d_type);

    if ((type_class == H5T_INTEGER || type_class == H5T_FLOAT))
    {
        // if (H5Tequal(d_type, H5T_NATIVE_HBOOL) > 0)
        // {
        //     res = HDF5GetValueT<bool>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_CHAR) > 0)
        // {
        //     res = HDF5GetValueT<char>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_SHORT) > 0)
        // {
        //     res = HDF5GetValueT<short>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_INT) > 0)
        // {
        //     res = HDF5GetValueT<int>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_LONG) > 0)
        // {
        //     res = HDF5GetValueT<double>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_LLONG) > 0)
        // {
        //     res = HDF5GetValueT<long long>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_UCHAR) > 0)
        // {
        //     res = HDF5GetValueT<unsigned char>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_USHORT) > 0)
        // {
        //     res = HDF5GetValueT<unsigned short>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_UINT) > 0)
        // {
        //     res = HDF5GetValueT<unsigned int>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_ULONG) > 0)
        // {
        //     res = HDF5GetValueT<unsigned long>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_ULLONG) > 0)
        // {
        //     res = HDF5GetValueT<unsigned long long>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_FLOAT) > 0)
        // {
        //     res = HDF5GetValueT<float>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_DOUBLE) > 0)
        // {
        //     res = HDF5GetValueT<double>(obj_id, d_type, d_space, is_attribute);
        // }
        // else if (H5Tequal(d_type, H5T_NATIVE_LDOUBLE) > 0)
        // {
        //     res = HDF5GetValueT<long double>(obj_id, d_type, d_space, is_attribute);
        // }
    }
    else if (type_class == H5T_ARRAY)
    {
        FIXME;
    }
    else if (type_class == H5T_STRING && is_attribute)
    {
        switch (H5Sget_simple_extent_type(d_space))
        {
        case H5S_SCALAR:
        {
            size_t sdims = H5Tget_size(d_type);
            char buffer[sdims + 1];
            auto m_type = H5Tcopy(H5T_C_S1);
            H5_ERROR(H5Tset_size(m_type, sdims));
            H5_ERROR(H5Aread(obj_id, m_type, buffer));
            H5_ERROR(H5Tclose(m_type));
            res.set_value<std::string>(std::string(buffer));
        }
        break;
        case H5S_SIMPLE:
        {
            hsize_t num = 0;
            H5_ERROR(H5Sget_simple_extent_dims(d_space, &num, nullptr));
            auto** buffer = new char*[num];
            auto m_type = H5Tcopy(H5T_C_S1);
            H5_ERROR(H5Tset_size(m_type, H5T_VARIABLE));
            H5_ERROR(H5Aread(obj_id, m_type, buffer));
            H5_ERROR(H5Tclose(m_type));
            auto& array = res.as_array();
            for (int i = 0; i < num; ++i)
            {
                array.push_back().set_value<std::string>(buffer[i]);
                delete buffer[i];
            }
            delete[] buffer;
        }
        break;
        default:
            break;
        }
    }
    else if (type_class == H5T_TIME)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_BITFIELD)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_REFERENCE)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_ENUM)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_VLEN)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_NO_CLASS)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_OPAQUE)
    {
        UNIMPLEMENTED;
    }
    else if (type_class == H5T_COMPOUND)
    {
        UNIMPLEMENTED;
    }

    H5Tclose(d_type);
    H5Sclose(d_space);
    return res;
}

void HDF5WriteArray(hid_t g_id, std::string const& key, const DataBlock& blk)
{
    bool is_exist = H5Lexists(g_id, key.c_str(), H5P_DEFAULT) != 0;
    //            H5Oexists_by_name(loc_id, key.c_str(), H5P_DEFAULT) != 0;
    H5O_info_t g_info;
    if (is_exist)
    {
        H5_ERROR(H5Oget_info_by_name(g_id, key.c_str(), &g_info, H5P_DEFAULT));
    }

    if (is_exist && g_info.type != H5O_TYPE_DATASET)
    {
        H5Ldelete(g_id, key.c_str(), H5P_DEFAULT);
        is_exist = false;
    }
    int ndims = blk.shape().size();

    int inner_lower[ndims];
    int inner_upper[ndims];
    int outer_lower[ndims];
    int outer_upper[ndims];

    // blk.GetIndexBox(inner_lower, inner_upper);
    // blk.GetShape(outer_lower, outer_upper);

    hsize_t m_shape[ndims];
    hsize_t m_start[ndims];
    hsize_t m_count[ndims];
    hsize_t m_stride[ndims];
    hsize_t m_block[ndims];

    if (blk.is_slow_first())
    {
        for (int i = 0; i < ndims; ++i)
        {
            m_shape[i] = static_cast<hsize_t>(outer_upper[i] - outer_lower[i]);
            m_start[i] = static_cast<hsize_t>(inner_lower[i] - outer_lower[i]);
            m_count[i] = static_cast<hsize_t>(inner_upper[i] - inner_lower[i]);
            m_stride[i] = static_cast<hsize_t>(1);
            m_block[i] = static_cast<hsize_t>(1);
        }
    }
    else
    {
        for (int i = 0; i < ndims; ++i)
        {
            m_shape[ndims - 1 - i] = static_cast<hsize_t>(outer_upper[i] - outer_lower[i]);
            m_start[ndims - 1 - i] = static_cast<hsize_t>(inner_lower[i] - outer_lower[i]);
            m_count[ndims - 1 - i] = static_cast<hsize_t>(inner_upper[i] - inner_lower[i]);
            m_stride[ndims - 1 - i] = static_cast<hsize_t>(1);
            m_block[ndims - 1 - i] = static_cast<hsize_t>(1);
        }
    }
    hid_t m_space = H5Screate_simple(ndims, &m_shape[0], nullptr);
    H5_ERROR(H5Sselect_hyperslab(m_space, H5S_SELECT_SET, &m_start[0], &m_stride[0], &m_count[0], &m_block[0]));
    hid_t f_space = H5Screate_simple(ndims, &m_count[0], nullptr);
    hid_t dset;
    hid_t d_type = GetHDF5DataType(blk.value_type_info());
    H5_ERROR(dset = H5Dcreate(g_id, key.c_str(), d_type, f_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, blk.data()));

    H5_ERROR(H5Dclose(dset));
    if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));
    if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));
}

size_t HDF5SetValue(hid_t g_id, const std::string& key, const Node& node)
{
    ASSERT(g_id > 0);

    size_t count = 0;
    if (key.empty())
    {
    }
    else if (H5Lexists(g_id, key.c_str(), H5P_DEFAULT) > 0)
    {
        RUNTIME_ERROR << "Can not rewrite exist dataset/group! [" << key << "]" << std::endl;
    }
    else if (H5Aexists(g_id, key.c_str()) > 0)
    {
        H5_ERROR(H5Adelete(g_id, key.c_str()));
    }

    node.visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& s_str) {
                auto m_type = H5Tcopy(H5T_C_S1);
                H5_ERROR(H5Tset_size(m_type, s_str.size()));
                H5_ERROR(H5Tset_strpad(m_type, H5T_STR_NULLTERM));
                auto m_space = H5Screate(H5S_SCALAR);
                auto aid = H5Acreate(g_id, key.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
                H5_ERROR(H5Awrite(aid, m_type, s_str.c_str()));
                H5_ERROR(H5Tclose(m_type));
                H5_ERROR(H5Sclose(m_space));
                H5_ERROR(H5Aclose(aid));
                ++count;
            },
            [&](const std::variant_alternative_t<Node::tags::Array, Node::value_type>& array) {
                std::vector<char const*> s_array;
                array->for_each([&](const Node&, const Node& v) {
                    s_array.push_back(v.as<Node::tags::String>().c_str());
                });

                hsize_t s = s_array.size();
                auto m_space = H5Screate_simple(1, &s, nullptr);
                auto m_type = H5Tcopy(H5T_C_S1);
                H5_ERROR(H5Tset_size(m_type, H5T_VARIABLE));
                auto aid = H5Acreate(g_id, key.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
                H5_ERROR(H5Awrite(aid, m_type, &s_array[0]));
                H5_ERROR(H5Tclose(m_type));
                H5_ERROR(H5Sclose(m_space));
                H5_ERROR(H5Aclose(aid));
                count = s;
            },
            [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>&) {
                NOT_IMPLEMENTED;
            },
            [&](const std::variant_alternative_t<Node::tags::Block, Node::value_type>& blk) {
                bool is_exist = H5Lexists(g_id, key.c_str(), H5P_DEFAULT) != 0;
                //            H5Oexists_by_name(loc_id, key.c_str(), H5P_DEFAULT) != 0;
                H5O_info_t g_info;
                if (is_exist)
                {
                    H5_ERROR(H5Oget_info_by_name(g_id, key.c_str(), &g_info, H5P_DEFAULT));
                }

                if (is_exist && g_info.type != H5O_TYPE_DATASET)
                {
                    H5Ldelete(g_id, key.c_str(), H5P_DEFAULT);
                    is_exist = false;
                }

                static constexpr int ndims = 3; // blk.GetNDIMS();

                int inner_lower[ndims];
                int inner_upper[ndims];
                int outer_lower[ndims];
                int outer_upper[ndims];

                // blk.GetIndexBox(inner_lower, inner_upper);
                // blk.GetIndexBox(outer_lower, outer_upper);

                hsize_t m_shape[ndims];
                hsize_t m_start[ndims];
                hsize_t m_count[ndims];
                hsize_t m_stride[ndims];
                hsize_t m_block[ndims];
                for (int i = 0; i < ndims; ++i)
                {
                    m_shape[i] = static_cast<hsize_t>(outer_upper[i] - outer_lower[i]);
                    m_start[i] = static_cast<hsize_t>(inner_lower[i] - outer_lower[i]);
                    m_count[i] = static_cast<hsize_t>(inner_upper[i] - inner_lower[i]);
                    m_stride[i] = static_cast<hsize_t>(1);
                    m_block[i] = static_cast<hsize_t>(1);
                }
                hid_t m_space = H5Screate_simple(ndims, &m_shape[0], nullptr);
                H5_ERROR(H5Sselect_hyperslab(m_space, H5S_SELECT_SET, &m_start[0], &m_stride[0], &m_count[0], &m_block[0]));
                hid_t f_space = H5Screate_simple(ndims, &m_count[0], nullptr);
                hid_t dset;
                hid_t d_type = GetHDF5DataType(blk.value_type_info());
                H5_ERROR(dset = H5Dcreate(g_id, key.c_str(), d_type, f_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, blk.data()));

                H5_ERROR(H5Dclose(dset));
                if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));
                if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));
                ++count;
            },
            [&](auto&& v) {
                NOT_IMPLEMENTED;
                //                 hid_t d_type = -1;
                //                 hid_t d_space;
                //                 count = 1;
                //                 // auto ndims = v.shape().size();
                //                 // if (ndims > 0)
                //                 // {
                //                 //     size_t d[ndims];
                //                 //     hsize_t h5d[ndims];
                //                 //     // entity->extents(d);
                //                 //     for (size_t i = 0; i < ndims; ++i)
                //                 //     {
                //                 //         h5d[i] = d[i];
                //                 //         count *= d[i];
                //                 //     }
                //                 //     d_space = H5Screate_simple(ndims, h5d, nullptr);
                //                 // }
                //                 // else
                //                 // {
                //                 //     d_space = H5Screate(H5S_SCALAR);
                //                 // }

                //                 if (false)
                //                 {
                //                 }
                // #define DEC_TYPE(_T_, _H5_T_)                      \
                //     else if (v.value_type_info() == typeid(_T_)) \
                //     {                                              \
                //         d_type = _H5_T_;                           \
                //     }
                //                 DEC_TYPE(bool, H5T_NATIVE_HBOOL)
                //                 DEC_TYPE(float, H5T_NATIVE_FLOAT)
                //                 DEC_TYPE(double, H5T_NATIVE_DOUBLE)
                //                 DEC_TYPE(int, H5T_NATIVE_INT)
                //                 DEC_TYPE(long, H5T_NATIVE_LONG)
                //                 DEC_TYPE(unsigned int, H5T_NATIVE_UINT)
                //                 DEC_TYPE(unsigned long, H5T_NATIVE_ULONG)
                // #undef DEC_TYPE
                //                 if (d_type != -1)
                //                 {
                //                     auto aid = H5Acreate(g_id, key.c_str(), d_type, d_space, H5P_DEFAULT, H5P_DEFAULT);
                //                     if (blk.is_continue())
                //                     {
                //                         H5_ERROR(H5Awrite(aid, d_type, blk.data()));
                //                     }
                //                     else
                //                     {
                //                         NOT_IMPLEMENTED;
                //                         // auto* ptr = operator new(blk.GetAlignOf());
                //                         // blk.CopyOut(ptr);
                //                         // H5_ERROR(H5Awrite(aid, d_type, ptr));
                //                         // operator delete(ptr);
                //                     }
                //                     H5_ERROR(H5Aclose(aid));
                //                 }
                //                 else
                //                 {
                //                     FIXME << "Can not write hdf5 attribute! " << std::endl
                //                           << key << " = " << node << std::endl;
                //                 }
                //                 H5_ERROR(H5Sclose(d_space));

                ++count;
            }

        });

    return count;
}

} // namespace _detail

struct hdf5_node
{
    std::shared_ptr<hid_t> m_fid_;
    hid_t m_gid_ = 0;
    std::string m_key_ = "";

    hdf5_node() : m_fid_(nullptr) {}

    ~hdf5_node() { close_group(); }

    auto open_group(const std::string& gpath, bool create_if_not_exist = true)
    {
        return _detail::H5GroupTryOpen(m_gid_ == 0 ? *m_fid_ : m_gid_, gpath, create_if_not_exist);
    }

    void open(const std::string& file, const std::string& grp = "/", std::string const& mode = "create")
    {
        close_all();
        if (mode == "create")
        {

            m_fid_ = std::shared_ptr<hid_t>(new hid_t(0), [&](hid_t* p) {
                if (p != nullptr && *p != 0)
                {
                    H5_ERROR(H5Fclose(*p));
                }
                delete p;
            });

            H5_ERROR(*m_fid_ = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
            VERBOSE << "Create HDF5 file: " << file;
        }

        std::tie(m_gid_, m_key_) = open_group(grp, mode == "create");
    }

    void close_all() { m_fid_.reset(); }

    void close_group()
    {
        if (m_gid_ != 0)
        {
            H5_ERROR(H5Gclose(m_gid_));
        }
    }

    Node update(const std::string& path, const Node& data)
    {
    }

    Node fetch(const std::string& path, Node const& projection) const
    {
        Node res;
        return std::move(res);
    }
};

typedef NodePlugin<hdf5_node> NodePluginHDF5;

template <>
void NodePluginHDF5::load(const Node& opt)
{
    opt.visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& path) {
                m_container_.open(path, "/", "create");
            },
            [&](const std::variant_alternative_t<Node::tags::Path, Node::value_type>& path) {
                m_container_.open(path.str(), "/", "create");
            },
            [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& object_p) {
                m_container_.open(object_p->find_child("file").get_value<std::string>("unnamed.h5"),
                                  object_p->find_child("path").get_value<std::string>("/"),
                                  object_p->find_child("mode").get_value<std::string>("create"));
            },
            [&](auto&& ele) { NOT_IMPLEMENTED; } //
        });
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

template <>
Node NodePluginHDF5::update(const Node& query, const Node& data, const Node& opt)
{
    Node res;

    query.visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& path) { m_container_.update(path, data).swap(res); },
            [&](const std::variant_alternative_t<Node::tags::Path, Node::value_type>& path) { m_container_.update(path.str(), data).swap(res); },
            [&](auto&& ele) { NOT_IMPLEMENTED; } //
        });

    return res;
}

template <>
Node NodePluginHDF5::fetch(const Node& query, const Node& projection, const Node& opt) const
{
    Node res;
    query.visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& path) { m_container_.fetch(path, projection).swap(res); },
            [&](const std::variant_alternative_t<Node::tags::Path, Node::value_type>& path) { m_container_.fetch(path.str(), projection).swap(res); },
            [&](auto&& ele) { NOT_IMPLEMENTED; } //
        });
    return res;
}

SPDB_ENTRY_REGISTER(hdf5, hdf5_node);
SPDB_ENTRY_ASSOCIATE(hdf5, hdf5_node, "^(.*)\\.(hdf5|h5)$");

} // namespace sp::db