//
// Created by salmon on 17-3-10.
//
#include "spdm/SpDMHDF5.h"
#include <sys/stat.h>
#include <fstream>
#include <regex>
#include <sstream>
#include <cassert>
extern "C" {
#include <hdf5.h>
#include <hdf5_hl.h>
}
#include "spdm/SpDM.h"
namespace sp {
namespace data {

#define H5_ERROR(_FUN_)                                                                                  \
    if ((_FUN_) < 0) {                                                                                   \
        H5Eprint(H5E_DEFAULT, stderr);                                                                   \
        throw(std::runtime_error(std::string("HDF5 Error:") + std::string(__FILE__) + std::string(":") + \
                                 std::to_string(__LINE__) + std::string(":") + __STRING(_FUN_)));        \
    }
struct H5Gcloser_s {
    void operator()(hid_t* p) const {
        if (p != nullptr && *p != -1) {
            H5Gclose(*p);
            delete p;
        }
    }
    explicit H5Gcloser_s(std::shared_ptr<hid_t> const& f) : m_file_(f) {}
    ~H5Gcloser_s() = default;
    std::shared_ptr<hid_t> m_file_;
};
struct H5Fcloser_s {
    void operator()(hid_t* p) const {
        if (p != nullptr && *p != -1) {
            H5Fclose(*p);
            delete p;
        }
    }
};

struct HDF5SAXWriter::pimpl_s {
    std::deque<std::shared_ptr<hid_t>> m_stack_;
    std::string m_key_ = "/";
    bool is_array = false;
    template <typename... Args>
    bool Put(Args&&... args);
    template <typename... Args>
    bool PutToArray(Args&&... args) {
        return true;
    };
    bool StartArray();
    bool EndArray();

    SpDataEntry m_array_cache_;
    std::deque<SpDataEntry*> m_array_stack_;
};
HDF5SAXWriter::HDF5SAXWriter(std::string const& path) : m_pimpl_(new pimpl_s) {
    hid_t f_id;
    H5_ERROR(f_id = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    m_pimpl_->m_stack_.push_front(std::shared_ptr<hid_t>(new hid_t(f_id), H5Fcloser_s()));
}
HDF5SAXWriter::~HDF5SAXWriter() { delete m_pimpl_; }
void HDF5SAXWriter::close() { m_pimpl_->m_stack_.clear(); }
template <typename V>
bool HDF5Put(hid_t g_id, std::string const& key, V const* d, hid_t d_type, unsigned int rank, size_t const* dims) {
    hid_t d_space;
    if (rank > 0) {
        hsize_t h5d[rank];
        for (size_t i = 0; i < rank; ++i) { h5d[i] = dims[i]; }
        H5_ERROR(d_space = H5Screate_simple(rank, h5d, nullptr));
    } else {
        H5_ERROR(d_space = H5Screate(H5S_SCALAR));
    }

    if (d_type == -1 || d_space == -1) { throw(std::runtime_error("Can not write hdf5 attribute! " + key)); }
    auto aid = H5Acreate(g_id, key.c_str(), d_type, d_space, H5P_DEFAULT, H5P_DEFAULT);
    H5_ERROR(H5Awrite(aid, d_type, d));
    H5_ERROR(H5Aclose(aid));
    H5_ERROR(H5Sclose(d_space));
    return true;
}
bool HDF5Put(hid_t g_id, std::string const& key, char const* str, size_t len = -1) {
    auto m_type = H5Tcopy(H5T_C_S1);
    len = (len <= 0) ? strlen(str) : len;
    H5_ERROR(H5Tset_size(m_type, len));
    H5_ERROR(H5Tset_strpad(m_type, H5T_STR_NULLTERM));
    auto m_space = H5Screate(H5S_SCALAR);
    auto aid = H5Acreate(g_id, key.c_str(), m_type, m_space, H5P_DEFAULT, H5P_DEFAULT);
    H5_ERROR(H5Awrite(aid, m_type, str));
    H5_ERROR(H5Tclose(m_type));
    H5_ERROR(H5Sclose(m_space));
    H5_ERROR(H5Aclose(aid));
    return true;
}
bool HDF5Put(hid_t g_id, std::string const& key, DataBlock const* v) {
    //    else if (auto p = std::dynamic_pointer_cast<const ArrayBase>(entity)) {
    //        if (auto m_data_ = p->pointer()) {
    //            bool is_exist = H5Lexists(g_id, key.c_str(), H5P_DEFAULT) != 0;
    //            //            H5Oexists_by_name(loc_id, key.c_str(), H5P_DEFAULT) != 0;
    //            H5O_info_t g_info;
    //            if (is_exist) { H5_ERROR(H5Oget_info_by_name(g_id, key.c_str(), &g_info, H5P_DEFAULT)); }
    //
    //            if (is_exist && g_info.type != H5O_TYPE_DATASET) {
    //                H5Ldelete(g_id, key.c_str(), H5P_DEFAULT);
    //                is_exist = false;
    //            }
    //            static constexpr int ndims = 3;  // p->GetNDIMS();
    //
    //            index_type inner_lower[ndims];
    //            index_type inner_upper[ndims];
    //            index_type outer_lower[ndims];
    //            index_type outer_upper[ndims];
    //
    //            p->GetIndexBox(inner_lower, inner_upper);
    //            p->GetIndexBox(outer_lower, outer_upper);
    //
    //            hsize_t m_shape[ndims];
    //            hsize_t m_start[ndims];
    //            hsize_t m_count[ndims];
    //            hsize_t m_stride[ndims];
    //            hsize_t m_block[ndims];
    //            for (int i = 0; i < ndims; ++i) {
    //                m_shape[i] = static_cast<hsize_t>(outer_upper[i] - outer_lower[i]);
    //                m_start[i] = static_cast<hsize_t>(inner_lower[i] - outer_lower[i]);
    //                m_count[i] = static_cast<hsize_t>(inner_upper[i] - inner_lower[i]);
    //                m_stride[i] = static_cast<hsize_t>(1);
    //                m_block[i] = static_cast<hsize_t>(1);
    //            }
    //            hid_t m_space = H5Screate_simple(ndims, &m_shape[0], nullptr);
    //            H5_ERROR(H5Sselect_hyperslab(m_space, H5S_SELECT_SET, &m_start[0], &m_stride[0], &m_count[0],
    //            &m_block[0]));
    //            hid_t f_space = H5Screate_simple(ndims, &m_count[0], nullptr);
    //            hid_t dset;
    //            hid_t d_type = GetHDF5DataType(p->value_type_info());
    //            H5_ERROR(dset = H5Dcreate(g_id, key.c_str(), d_type, f_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    //            H5_ERROR(H5Dwrite(dset, d_type, m_space, f_space, H5P_DEFAULT, m_data_));
    //
    //            H5_ERROR(H5Dclose(dset));
    //            if (m_space != H5S_ALL) H5_ERROR(H5Sclose(m_space));
    //            if (f_space != H5S_ALL) H5_ERROR(H5Sclose(f_space));
    //            ++m_count_;
    //        }
    //    }
    return true;
}
bool HDF5Put(hid_t g_id, std::string const& key, H5T_class_t d_type) {
    auto aid = H5Acreate(g_id, key.c_str(), d_type, H5S_SCALAR, H5P_DEFAULT, H5P_DEFAULT);
    H5_ERROR(H5Aclose(aid));
    return true;
}
template <typename... Args>
bool HDF5SAXWriter::pimpl_s::Put(Args&&... args) {
    if (is_array) { return PutToArray(std::forward<Args>(args)...); }
    assert(!m_key_.empty());
    bool success;
    auto g_id = **m_stack_.begin();

    if (m_key_.empty()) {
    } else if (H5Lexists(g_id, m_key_.c_str(), H5P_DEFAULT) > 0) {
        throw(std::runtime_error("Can not rewrite exist dataset/group! [" + m_key_ + "]"));
    } else if (H5Aexists(g_id, m_key_.c_str()) > 0) {
        H5_ERROR(H5Adelete(g_id, m_key_.c_str()));
    }
    success = HDF5Put(g_id, m_key_, std::forward<Args>(args)...);
    return success;
}
bool HDF5SAXWriter::Null() { return m_pimpl_->Put(H5T_NCLASSES); }
bool HDF5SAXWriter::Bool(bool b) { return TensorBool(&b, 0, nullptr); };
bool HDF5SAXWriter::Int(int i) { return TensorInt(&i, 0, nullptr); };
bool HDF5SAXWriter::Uint(unsigned u) { return TensorUint(&u, 0, nullptr); };
bool HDF5SAXWriter::Int64(int64_t i) { return TensorInt64(&i, 0, nullptr); };
bool HDF5SAXWriter::Uint64(uint64_t u) { return TensorUint64(&u, 0, nullptr); };
bool HDF5SAXWriter::Double(double d) { return TensorDouble(&d, 0, nullptr); };
bool HDF5SAXWriter::String(const char* str, size_type length, bool copy) { return m_pimpl_->Put(str, length); }

bool HDF5SAXWriter::TensorBool(bool const* b, unsigned int rank, size_t const* dims) {
    return m_pimpl_->Put(b, H5T_NATIVE_HBOOL, rank, dims);
}
bool HDF5SAXWriter::TensorInt(int const* i, unsigned int rank, size_t const* dims) {
    return m_pimpl_->Put(i, H5T_NATIVE_INT, rank, dims);
}
bool HDF5SAXWriter::TensorUint(unsigned int const* u, unsigned int rank, size_t const* dims) {
    return m_pimpl_->Put(u, H5T_NATIVE_UINT, rank, dims);
}
bool HDF5SAXWriter::TensorInt64(int64_t const* i, unsigned int rank, size_t const* dims) {
    return m_pimpl_->Put(i, H5T_NATIVE_INT64, rank, dims);
}
bool HDF5SAXWriter::TensorUint64(uint64_t const* u, unsigned int rank, size_t const* dims) {
    return m_pimpl_->Put(u, H5T_NATIVE_UINT64, rank, dims);
}
bool HDF5SAXWriter::TensorDouble(double const* f, unsigned int rank, size_t const* dims) {
    return m_pimpl_->Put(f, H5T_NATIVE_DOUBLE, rank, dims);
}
// bool HDF5SAXWriter::Object(spObject const* obj) {
//    if (auto const* blk = dynamic_cast<DataBlock const*>(obj)) {
//        return m_pimpl_->Put(blk);
//    } else {
//        return false;
//    }
//}

bool HDF5SAXWriter::StartObject() {
    assert(!m_pimpl_->m_key_.empty());
    hid_t g_id, f_id;
    f_id = **m_pimpl_->m_stack_.begin();

    if (H5Lexists(f_id, m_pimpl_->m_key_.c_str(), H5P_DEFAULT) > 0) {
        H5O_info_t o_info;
        H5_ERROR(H5Oget_info_by_name(f_id, m_pimpl_->m_key_.c_str(), &o_info, H5P_DEFAULT));
        if (o_info.type == H5O_TYPE_GROUP) {
            H5_ERROR(g_id = H5Gopen(f_id, m_pimpl_->m_key_.c_str(), H5P_DEFAULT));
        } else {
            throw(std::runtime_error(m_pimpl_->m_key_ + " is  a  dataset!"));
        }
    } else if (H5Aexists(f_id, m_pimpl_->m_key_.c_str()) > 0) {
        throw(std::runtime_error(m_pimpl_->m_key_ + " is   an attribute!"));
    } else {
        H5_ERROR(g_id = H5Gcreate(f_id, m_pimpl_->m_key_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    }
    m_pimpl_->m_stack_.push_front(std::shared_ptr<hid_t>(new hid_t(g_id), H5Gcloser_s(*m_pimpl_->m_stack_.begin())));
    return true;
}
bool HDF5SAXWriter::Key(const char* str, size_type length, bool copy) {
    m_pimpl_->m_key_ = str;
    return true;
}
bool HDF5SAXWriter::EndObject(size_type memberCount) {
    m_pimpl_->m_stack_.pop_front();
    return true;
}
bool HDF5SAXWriter::StartArray() {
    //    m_pimpl_->m_array_stack_.push_front((*m_pimpl_->m_array_stack_.begin())->Insert(SpDM(), SpDM(kArray), true));
    m_pimpl_->is_array = true;
    return true;
}
bool HDF5SAXWriter::EndArray(size_type elementCount) {
    m_pimpl_->is_array = false;

    //    m_pimpl_->m_array_stack_.pop_front();
    //    if (m_pimpl_->m_array_stack_.empty()) {
    //        // Write array
    //        m_pimpl_->m_array_cache_.reset();
    //        m_pimpl_->is_array = false;
    //    }
    return true;
}
struct HDF5SAXReader::pimpl_s {
    std::string m_path_;
};
HDF5SAXReader::HDF5SAXReader() : m_pimpl_(new pimpl_s) {}
HDF5SAXReader::~HDF5SAXReader() { delete m_pimpl_; }

herr_t attr_op_func(hid_t loc_id /*in*/, const char* name /*in*/, const H5A_info_t* ainfo /*in*/,
                    void* op_data /*in,out*/) {
    auto* handler = reinterpret_cast<SpDMSAXInterface*>(op_data);

    auto attr = H5Aopen_name(loc_id, name);
    handler->Key(name);

    auto a_type = H5Aget_type(attr);
    auto a_space = H5Aget_space(attr);
    unsigned int rank = H5Sget_simple_extent_ndims(a_space);
    hsize_t sdim[64];
    H5_ERROR(H5Sget_simple_extent_dims(a_space, sdim, nullptr));

    auto npoints = H5Sget_simple_extent_npoints(a_space);
    switch (H5Tget_class(a_type)) {
        case H5T_INTEGER:
        case H5T_FLOAT:
            if (H5Tequal(a_type, H5T_NATIVE_HBOOL) > 0) {
                auto d = operator new(sizeof(bool) * npoints);
                H5_ERROR(H5Aread(attr, a_type, d));
                handler->Value(reinterpret_cast<bool*>(d), rank, sdim);
                operator delete(d);
            } else if (H5Tequal(a_type, H5T_NATIVE_INT) > 0) {
                auto d = operator new(sizeof(int) * npoints);
                H5_ERROR(H5Aread(attr, a_type, d));
                handler->Value(reinterpret_cast<int*>(d), rank, sdim);
                operator delete(d);
            } else if (H5Tequal(a_type, H5T_NATIVE_UINT) > 0) {
                auto d = operator new(sizeof(unsigned int) * npoints);
                H5_ERROR(H5Aread(attr, a_type, d));
                handler->Value(reinterpret_cast<unsigned int*>(d), rank, sdim);
                operator delete(d);
            } else if (H5Tequal(a_type, H5T_NATIVE_INT64) > 0) {
                auto d = operator new(sizeof(int64_t) * npoints);
                H5_ERROR(H5Aread(attr, a_type, d));
                handler->Value(reinterpret_cast<int64_t*>(d), rank, sdim);
                operator delete(d);
            } else if (H5Tequal(a_type, H5T_NATIVE_UINT64) > 0) {
                auto d = operator new(sizeof(uint64_t) * npoints);
                H5_ERROR(H5Aread(attr, a_type, d));
                handler->Value(reinterpret_cast<uint64_t*>(d), rank, sdim);
                operator delete(d);
            } else if (H5Tequal(a_type, H5T_NATIVE_DOUBLE) > 0) {
                auto d = operator new(sizeof(double) * npoints);
                H5_ERROR(H5Aread(attr, a_type, d));
                handler->Value(reinterpret_cast<double*>(d), rank, sdim);
                operator delete(d);
            } else {
                handler->Null();
            }
            break;
        case H5T_STRING:
            switch (H5Sget_simple_extent_type(a_space)) {
                case H5S_SCALAR: {
                    auto len = H5Tget_size(a_type);
                    char buffer[len + 1];
                    auto m_type = H5Tcopy(H5T_C_S1);
                    H5_ERROR(H5Tset_size(m_type, len));
                    H5_ERROR(H5Aread(attr, m_type, buffer));
                    H5_ERROR(H5Tclose(m_type));
                    handler->String(buffer, len, true);
                } break;
                case H5S_SIMPLE: {
                    hsize_t num = 0;
                    H5_ERROR(H5Sget_simple_extent_dims(a_space, &num, nullptr));
                    auto** buffer = new char*[num];
                    auto m_type = H5Tcopy(H5T_C_S1);
                    H5_ERROR(H5Tset_size(m_type, H5T_VARIABLE));
                    H5_ERROR(H5Aread(attr, m_type, buffer));
                    H5_ERROR(H5Tclose(m_type));
                    handler->StartArray();
                    for (int i = 0; i < num; ++i) { handler->String(buffer[i], num); }
                    delete[] buffer;
                    handler->EndArray(num);

                } break;
                default:
                    break;
            }
            break;
        case H5T_ARRAY:
        case H5T_TIME:
        case H5T_BITFIELD:
        case H5T_REFERENCE:
        case H5T_ENUM:
        case H5T_VLEN:
        case H5T_NO_CLASS:
        case H5T_OPAQUE:
        case H5T_COMPOUND:
        case H5T_NCLASSES:
        default:
            handler->Null();
            break;
    }
    H5_ERROR(H5Tclose(a_type));
    H5_ERROR(H5Sclose(a_space));
    H5_ERROR(H5Aclose(attr));

    return 0;
}

struct grp_op_data_s {
    unsigned recurs;            /* Recursion level.  0=root */
    struct grp_op_data_s* prev; /* Pointer to previous grp_op_data_s */
    haddr_t addr;               /* Group address */
    SpDMSAXInterface* handler;
};
DataBlock* h5_open_dataset(hid_t loc_id, const char* path) {
    DataBlock* res = nullptr;
    return res;
}
/************************************************************

   This function recursively searches the linked list of
   opdata structures for one whose address matches
   target_addr.  Returns 1 if a match is found, and 0
   otherwise.

  ************************************************************/
int group_check(grp_op_data_s* od, haddr_t target_addr) {
    if (od->addr == target_addr)
        return 1; /* Addresses match */
    else if (!od->recurs)
        return 0; /* Root group reached with no matches */
    else
        return group_check(od->prev, target_addr);
    /* Recursively examine the next node */
}
/************************************************************

  Operator function.  This function prints the name and type
  of the object passed to it.  If the object is a group, it
  is first checked against other groups in its path using
  the group_check function, then if it is not a duplicate,
  H5Literate is called for that group.  This guarantees that
  the program will not enter infinite recursion due to a
  circular path in the file.

 ************************************************************/
herr_t grp_op_func(hid_t loc_id, const char* name, const H5L_info_t* info, void* operator_data) {
    H5O_info_t infobuf;
    H5_ERROR(H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT));

    auto* od = reinterpret_cast<grp_op_data_s*>(operator_data);
    od->handler->StartObject();
    if (!(name[0] == '/' && name[1] == '\0')) {
        od->handler->Key("@name");
        od->handler->String(name);
    }
    H5G_info_t g_info;
    H5_ERROR(H5Gget_info(loc_id, &g_info));

    //    std::cerr << infobuf.num_attrs << "/" << g_info.nlinks << std::endl;
    if (infobuf.num_attrs > 0) {
        H5_ERROR(H5Aiterate_by_name(loc_id, name, H5_INDEX_NAME, H5_ITER_INC, nullptr, attr_op_func,
                                    reinterpret_cast<void*>(od->handler), H5P_DEFAULT));
    }
    if (g_info.nlinks > infobuf.num_attrs) {
        od->handler->Key("Group");
        od->handler->StartArray();
        int count = 0;
        /*
         * Get type of the tree and display its name and type.
         * The name of the tree is passed to this function by
         * the Library.
         */
        switch (infobuf.type) {
            case H5O_TYPE_GROUP: {
                /*
                 * Check group address against linked list of operator
                 *  data structures.  We will always run the check, as the
                 * reference m_count_ cannot be relied upon if there are
                 * symbolic links, and H5Oget_info_by_name always follows
                 * symbolic links.  Alternatively we could use H5Lget_info
                 * and never recurse on groups discovered by symbolic
                 * links, however it could still fail if an tree's
                 * reference m_count_ was manually manipulated with
                 * H5Odecr_refcount.
                 */
                if (group_check(od, infobuf.addr)) {
                    throw(std::runtime_error(" Warning: Loop detected!"));
                } else {
                    ++count;
                    /*
                     * Initialize new operator m_data_ structure and
                     * begin recursive iteration on the discovered
                     * group.  The new grp_op_data_s structure is given a
                     * pointer to the current one.
                     */
                    grp_op_data_s nextod{
                        .recurs = od->recurs + 1, .prev = od, .addr = infobuf.addr, .handler = od->handler};
                    H5_ERROR(H5Literate_by_name(loc_id, name, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr, grp_op_func,
                                                (void*)&nextod, H5P_DEFAULT));
                }

            } break;
            case H5O_TYPE_DATASET:
                //            od->handler->Object(h5_open_dataset(loc_id, name));
                break;
            case H5O_TYPE_NAMED_DATATYPE:
                break;
            default:
                break;
        }
        od->handler->EndArray(count);
    }

    od->handler->EndObject(1);

    return 0;
}

void HDF5SAXReader::Parse(std::string const& path, sp::data::SpDMSAXInterface& handler) const {
    hid_t f_id;
    H5_ERROR(f_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    grp_op_data_s od{.recurs = 0, .prev = nullptr, .addr = 0, .handler = &handler};
    grp_op_func(f_id, "/", nullptr, reinterpret_cast<void*>(&od));
    H5_ERROR(H5Fclose(f_id));
}
}  // namespace m_data_{
}  // namespace sp{