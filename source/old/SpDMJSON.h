//
// Created by salmon on 18-2-9.
//

#ifndef SIMPLA_SPDMJSON_H
#define SIMPLA_SPDMJSON_H

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include "SpDM.h"
#include "SpDMSAX.h"

namespace simpla {
namespace traits {
template <typename StringBuffer>
struct is_sax_interface<rapidjson::PrettyWriter<StringBuffer>> : public std::true_type {};
}
// namespace traits {

template <typename DOM = SpDataEntry>
DOM ReadJSON(char const *json, std::size_t len = 0) {
    rapidjson::Reader reader;
    rapidjson::InsituStringStream ss(const_cast<char *>(json));
    DOM db;
    SpDMSAXWrapper<DOM> writer(db);
    reader.Parse(ss, writer);
    return std::move(db);
}
template <typename IS, typename DOM = SpDataEntry>
auto ReadJSON(IS &input_stream) {
    std::string json(std::istreambuf_iterator<char>(input_stream), {});
    return ReadJSON<DOM>(json.c_str(), json.size());
}
template <typename OS, typename DOM>
OS &WriteJSON(OS &output_stream, DOM const &db) {
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    db.Serialize(writer);
    output_stream << buffer.GetString();
    return output_stream;
}


inline SpDMElement<> operator"" _json(const char *c, std::size_t n) {
    return ReadJSON<SpDMElement<>>(c, n);
}
}  // namespace simpla
#endif  // SIMPLA_SPDMJSON_H
