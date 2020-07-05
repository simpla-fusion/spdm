//
// Created by salmon on 18-2-9.
//

#ifndef SIMPLA_SPDMHDF5_H
#define SIMPLA_SPDMHDF5_H

#include <string>
#include "SpDM.h"
#include "SpDMFactory.h"

namespace sp {
namespace data {
class DataBlock;
struct HDF5SAXWriter : public SpDMSAXInterface {
    typedef size_t size_type;
    HDF5SAXWriter(std::string const &path);
    virtual ~HDF5SAXWriter();

    void close();

    bool Null();
    bool Bool(bool b);
    bool Int(int i);
    bool Uint(unsigned u);
    bool Int64(int64_t i);
    bool Uint64(uint64_t u);
    bool Double(double d);
    bool String(const char *str, size_type length, bool copy);

    bool StartObject();
    bool Key(const char *str, size_type length, bool copy);
    bool EndObject(size_type memberCount);
    bool StartArray();
    bool EndArray(size_type elementCount);

    bool TensorBool(bool const *b, unsigned int rank, size_t const *dims);
    bool TensorInt(int const *i, unsigned int rank, size_t const *dims);
    bool TensorUint(unsigned const *u, unsigned int rank, size_t const *dims);
    bool TensorInt64(int64_t const *i, unsigned int rank, size_t const *dims);
    bool TensorUint64(uint64_t const *u, unsigned int rank, size_t const *dims);
    bool TensorDouble(double const *d, unsigned int rank, size_t const *dims);
    //    bool Object(spObject const *);

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

struct HDF5SAXReader {
    HDF5SAXReader();
    ~HDF5SAXReader();

    void Parse(std::string const &path, SpDMSAXInterface &) const;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

template <typename DOM>
void ReadHDF5(DOM &db, std::string const &path) {
    HDF5SAXReader reader;
    SpDMFactory<DOM> deserializer(db);
    reader.Parse(path, deserializer);
}
template <typename DOM>
DOM *DeserializeFromHDF5(std::string const &path) {
    auto *db = new DOM;
    ReadHDF5(*db, path);
    return db;
}

template <typename DOM>
void WriteHDF5(std::string const &path, DOM const &db) {
    HDF5SAXWriter writer(path);
    db.Serialize(writer);
};
}  //    namespace m_data_{
}  // namespace sp{

#endif  // SIMPLA_SPDMHDF5_H
