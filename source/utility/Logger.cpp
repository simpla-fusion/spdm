/**
 *  @file  logger.cpp
 *
 * @date    2014-7-29  AM8:43:27
 * @author salmon
 * 
 * * change (20200713):
 *      - filename => logger.cpp
 */

#include "Logger.h"
#include "Singleton.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace sp
{
namespace logger
{

/**
         *  @ingroup Logging
         *  \brief Logging stream, should be used  as a singleton
         */
struct LoggerStreams
{
    static constexpr unsigned int DEFAULT_LINE_WIDTH = 120;
    bool is_opened_ = false;
    int line_width_ = DEFAULT_LINE_WIDTH;
    int mpi_rank_ = 0, mpi_size_ = 1;

    LoggerStreams(int level = LOG_INFORM) : m_std_out_level_(level), line_width_(DEFAULT_LINE_WIDTH) {}
    ~LoggerStreams() { close(); }

    int init();

    int close();

    int open_file(std::string const& name)
    {
        if (fs.is_open())
            fs.close();
        fs.open(name.c_str(), std::ios_base::trunc);
        return 0;
    }

    int push(int level, std::string const& msg);

    int set_stdout_level(int l)
    {
        m_std_out_level_ = l;
        return l;
    }

    int get_line_width() const { return line_width_; }

    int set_line_width(int lineWidth)
    {
        line_width_ = lineWidth;
        return lineWidth;
    }

    static std::string time_stamp()
    {
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

        char mtstr[100];
        std::strftime(mtstr, 100, "%F %T", std::localtime(&now));

        return std::string(mtstr);
    }

private:
#ifndef NDEBUG
    int m_std_out_level_ = 1000;
#else
    int m_std_out_level_ = 0;
#endif
    std::ofstream fs;
};

int LoggerStreams::init()
{
    is_opened_ = true;
    return 0;
}

int LoggerStreams::close()
{
    if (is_opened_)
    {
        VERBOSE << "LoggerStream is closed!" << std::endl;
        if (m_std_out_level_ >= LOG_INFORM && mpi_rank_ == 0)
        {
            std::cout << std::endl;
        }
        if (fs.is_open())
        {
            fs.close();
        }
        is_opened_ = false;
    }
    return 0;
}

int LoggerStreams::push(int level, std::string const& msg)
{
    if (msg == "" || ((level == LOG_MESSAGE) && mpi_rank_ > 0))
        return 0;

    std::ostringstream prefix;

    std::string surfix;

    switch (level)
    {
    case LOG_FORCE_OUTPUT:
    case LOG_OUT_RANGE_ERROR:
    case LOG_LOGIC_ERROR:
    case LOG_ERROR:
        prefix << "[E]";
        break;
    case LOG_WARNING:
        prefix << "[W]"; // red
        break;
    case LOG_LOG:
        prefix << "[L]";
        break;
    case LOG_VERBOSE:
        prefix << "[V]";
        break;
    case LOG_INFORM:
        prefix << "[I]";
        break;
    case LOG_DEBUG:
        prefix << "[D]";
        break;
    default:
        break;
    }

    prefix << "[" << time_stamp() << "] ";

    if (mpi_size_ > 1)
    {
        prefix << "[" << mpi_rank_ << "/" << mpi_size_ << "] ";
    }

    if (level >= m_std_out_level_)
    {
        switch (level)
        {
        case LOG_FORCE_OUTPUT:
        case LOG_OUT_RANGE_ERROR:
        case LOG_LOGIC_ERROR:
        case LOG_ERROR:
        case LOG_ERROR_RUNTIME:
        case LOG_ERROR_BAD_CAST:
        case LOG_ERROR_LOGICAL:
            std::cerr << std::setw(30) << std::left << prefix.str() << "\e[91m" << msg << surfix << "\e[0m"
                      << std::endl;
            break;
        case LOG_WARNING:
            std::cerr << std::setw(30) << std::left << prefix.str() << "\e[96m" << msg << surfix << "\e[0m"
                      << std::endl;
            break;
        case LOG_MESSAGE:
            std::cout << msg << "\e[0m" << std::endl;
            break;
        default:
            std::cout << std::setw(30) << std::left << prefix.str() << msg << surfix << "\e[0m" << std::endl;
        }
    }

    if (fs.good())
    {
        fs << std::endl
           << prefix.str() << msg << surfix;
    }
    return 0;
}

int open_file(std::string const& file_name) { return Singleton<LoggerStreams>::instance().open_file(file_name); }

int close() { return sp::Singleton<LoggerStreams>::instance().close(); }

int set_stdout_level(int l) { return Singleton<LoggerStreams>::instance().set_stdout_level(l); }

int set_mpi_comm(int r, int s)
{
    Singleton<LoggerStreams>::instance().mpi_rank_ = r;
    Singleton<LoggerStreams>::instance().mpi_size_ = s;
    return 0;
}

int set_line_width(int lw) { return Singleton<LoggerStreams>::instance().set_line_width(lw); }

int get_line_width() { return Singleton<LoggerStreams>::instance().get_line_width(); }

Logger::Logger() : base_type(), m_level_(0), current_line_char_count_(0), endl_(true) {}

Logger::Logger(int lv) : m_level_(lv), current_line_char_count_(0), endl_(true)
{
    base_type::operator<<(std::boolalpha);

    current_line_char_count_ = get_buffer_length();
}

Logger::~Logger()
{
    flush();
    // switch (m_level_)
    // {
    // case LOG_ERROR_RUNTIME:
    //     throw(std::runtime_error(this->str()));
    // case LOG_ERROR_BAD_CAST:
    //     flush();
    //     throw(std::bad_cast());
    // case LOG_ERROR_OUT_OF_RANGE:
    //     throw(std::out_of_range(this->str()));
    // case LOG_ERROR_LOGICAL:
    //     throw(std::logic_error(this->str()));
    // case LOG_ERROR_DOMAIN:
    //     throw(std::domain_error(this->str()));
    // case LOG_ERROR_INVALID_ARGUMENT:
    //     throw(std::invalid_argument(this->str()));
    // default:
    //     break;
    // }
}

int Logger::get_buffer_length() const { return static_cast<int>(this->str().size()); }

void Logger::flush()
{
    Singleton<LoggerStreams>::instance().push(m_level_, this->str());
    this->str("");
}

void Logger::surffix(std::string const& s)
{
    (*this) << std::setfill('.')
            << std::setw(Singleton<LoggerStreams>::instance().get_line_width() - current_line_char_count_)
            << std::right << s << std::left << std::endl;

    flush();
}

void Logger::endl()
{
    (*this) << std::endl;
    current_line_char_count_ = 0;
    endl_ = true;
}

void Logger::not_endl() { endl_ = false; }

} // namespace logger
} // namespace sp
// namespace simpla
