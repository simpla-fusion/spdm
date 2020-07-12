#include "XPath.h"
#include "utility/Logger.h"
#include "utility/URL.h"
using namespace sp;
XPath::XPath(const std::string &path) : m_path_(path) {}
// XPath::~XPath() = default;
// XPath::XPath(XPath &&) = default;
// XPath::XPath(XPath const &) = default;
// XPath &XPath::operator=(XPath const &) = default;
const std::string &XPath::str() const { return m_path_; }

XPath XPath::operator/(const std::string &suffix) const { return XPath(urljoin(m_path_, suffix)); }
XPath::operator std::string() const { return m_path_; }
