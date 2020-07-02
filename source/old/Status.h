//
// Created by salmon on 18-4-18.
//

#ifndef SIMPLA_SPDMSTATUS_H
#define SIMPLA_SPDMSTATUS_H

#include <utility>
namespace simpla {

struct Status {
    /**
     * @ref http://www.iana.org/assignments/http-status-codes/http-status-codes.xhtml
     */
    enum status_code {
        kInformational = 100,
        kContinue = 100,
        kSuccessful = 200,
        kOK = 200,
        kCreated = 201,
        kAccepted = 202,
        kRedirection = 300,
        kMovedPermanently = 301,
        kNotModified = 304,
        kClientError = 400,
        kUnauthorized = 401,
        kPaymentRequired = 402,
        kNotFound = 404,
        kMethodNotAllowed = 405,
        kServerError = 500,
        kInternalServerError = 500,
        kNotImplemented = 501
    };

    Status(int c = kOK) : m_code_(c) {}
    Status(Status const& other) : m_code_(other.m_code_) {}
    ~Status() = default;
    Status& operator=(Status const& other) {
        Status(other).swap(*this);
        return *this;
    }
    void swap(Status& other) { std::swap(m_code_, other.m_code_); }

    int code() const { return m_code_; }
    operator int() const { return m_code_; }

    bool operator==(Status const& other) { return m_code_ == other.m_code_; }
    bool operator!=(Status const& other) { return m_code_ != other.m_code_; }

    static Status OK() { return Status(kOK); }
    static Status NotModified() { return Status(kNotModified); }

   private:
    int m_code_ = kOK;
};
}  // namespace simpla
#endif  // SIMPLA_SPDMSTATUS_H
