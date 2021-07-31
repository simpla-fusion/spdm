/**
 *   ____  _           ____  _
 * / ___|(_)_ __ ___ |  _ \| | __ _
 * \___ \| | '_ ` _ \| |_) | |/ _` |
 *  ___) | | | | | | |  __/| | (_| |
 * |____/|_|_| |_| |_|_|   |_|\__,_|
 *
 *
 *
 *
 * @file Logger.h
 *
 *  created on: 2012-3-21
 *      Author: salmon
 *  * change (20200713): 
 *      - filename => Logger.h
 *      - namespace => sp::logger
 *         
 */

#ifndef SP_LOGGER_H_
#define SP_LOGGER_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <stddef.h>
#include <string>

namespace sp
{
/** @ingroup utility */
namespace logger
{

int open_file(std::string const& file_name);

int close();

int set_stdout_level(int l);

int set_line_width(int lw);

int get_line_width();

int set_mpi_comm(int rank = 0, int size = 1);

/**
         * @ingroup utility
         * @addtogroup logging   Log
         * @{
         *
         */

enum tags
{
    LOG_FORCE_OUTPUT = -10000,    //!< LOG_FORCE_OUTPUT
    LOG_MESSAGE = -20,            //!< LOG_MESSAGE
    LOG_OUT_RANGE_ERROR = -4,     //!< LOG_OUT_RANGE_ERROR
    LOG_LOGIC_ERROR = -3,         //!< LOG_LOGIC_ERROR
    LOG_ERROR = -2,               //!< LOG_ERROR
    LOG_ERROR_RUNTIME = -10,      //!< LOG_ERROR_RUNTIME
    LOG_ERROR_BAD_CAST = -11,     //!< LOG_ERROR_RUNTIME
    LOG_ERROR_OUT_OF_RANGE = -12, //!< LOG_ERROR_RUNTIME
    LOG_ERROR_LOGICAL = -13,      //!< LOG_ERROR_RUNTIME
    LOG_ERROR_DOMAIN = -14,
    LOG_ERROR_INVALID_ARGUMENT = -15,
    LOG_ERROR_NOT_IMEPLEMENT = -16,

    LOG_WARNING = -1, //!< LOG_WARNING

    LOG_INFORM = 0, //!< LOG_INFORM
    LOG_LOG = 1,    //!< LOG_LOG

    LOG_VERBOSE = 10, //!< LOG_VERBOSE
    LOG_DEBUG = -30   //!< LOG_DEBUG
};
// CHECK_MEMBER_FUNCTION(has_member_function_print, print);

/**
         *
         *  @brief log message m_buffer,
         */
class Logger : public std::ostringstream
{
    typedef std::ostringstream base_type;
    typedef Logger this_type;

public:
    Logger();
    Logger(int lv);
    ~Logger();
    int get_buffer_length() const;
    void flush();
    void surffix(std::string const& s);
    void endl();
    void not_endl();

private:
public:
    template <typename T>
    inline this_type& push(
        T const& value
        //            , std::enable_if_t<!has_member_function_print<T, std::ostream &>::value> *__p = nullptr
    )
    {
        current_line_char_count_ -= get_buffer_length();
        *dynamic_cast<base_type*>(this) << (value);
        current_line_char_count_ += get_buffer_length();
        if (current_line_char_count_ > get_line_width())
        {
            endl();
        }
        return *this;
    }

    //    template <typename T>
    //    inline this_type &push(T const &value,
    //                           std::enable_if_t<has_member_function_print<T, std::ostream &>::value> *__p = nullptr) {
    //        current_line_char_count_ -= get_buffer_length();
    //        value.print(*dynamic_cast<base_type *>(this));
    //        current_line_char_count_ += get_buffer_length();
    //        if (current_line_char_count_ > get_line_width()) { endl(); }
    //
    //        return *this;
    //    }

    template <typename T>
    inline this_type const& push(T const& value) const
    {
        const_cast<this_type&>(*this).push(value);
        return *this;
    }

    typedef Logger& (*LoggerStreamManipulator)(Logger&);

    Logger& push(LoggerStreamManipulator manip)
    {
        // call the function, and return it's entity
        return manip(*this);
    }

    /**
     *
     * define the custom endl for this stream.
     * note how it matches the `LoggerStreamManipulator`
     * function signature
     *
     * 	static this_type& endl(this_type& stream)
     * {
     * 	// print a new line
     * 	std::cout << std::endl;
     *
     * 	// do other stuff with the stream
     * 	// std::cout, for example, will flush the stream
     * 	stream << "Called Logger::endl!" << std::endl;
     *
     * 	return stream;
     * }
     *
     *
     *
     *
     */

    int m_level_ = -10000;
    int current_line_char_count_;
    bool endl_;
};

// this is the function signature of std::endl
typedef std::basic_ostream<char, std::char_traits<char>> StdCoutType;
typedef StdCoutType& (*StandardEndLine)(StdCoutType&);
//! define an operator<< to take in std::endl
inline Logger& operator<<(Logger& self, StandardEndLine manip)
{
    // call the function, but we cannot return it's entity
    manip(dynamic_cast<std::ostringstream&>(self));
    return self;
}

inline Logger& operator<<(Logger& self, const char* arg) { return self.push(arg); }

template <typename Arg>
inline Logger& operator<<(Logger& self, Arg const& arg)
{
    return self.push(arg);
}

// template<typename Arg>
// Logger &operator<<(Logger &L, Arg const &arg)
//{
//    return L.push(arg);
//}
//
// template<typename Arg>
// Logger const &operator<<(Logger const &L, Arg const &arg)
//{
//    return L.push(arg);
//}
//
// inline Logger &operator<<(Logger &L, std::string const &arg)
//{
//    return L.push(arg);
//}
//
// inline Logger const &operator<<(Logger const &L, std::string const &arg)
//{
//    return L.push(arg);
//}

/**
         * @name     manip for Logger
         * @{
         **/

inline Logger& endl(Logger& self)
{
    self << std::endl;
    self.flush();
    return self;
}

inline Logger& done(Logger& self)
{
    self.surffix("[DONE]");
    return self;
}

inline Logger& failed(Logger& self)
{
    self.surffix("\e[31;1m[FAILED]\e[0m");
    return self;
}

inline Logger& start(Logger& self)
{
    self.surffix("[START]");
    return self;
}

inline Logger& flush(Logger& self)
{
    self.flush();
    return self;
}

// inline std::string ShowBit(unsigned long s) { return std::bitset<64>(s).to_string(); }

inline std::ostringstream& _make_error_msg(std::ostringstream& os) { return os; }

template <typename T>
std::ostringstream& _make_msg(std::ostringstream& os, T const& first)
{
    os << first;
    return os;
}

template <typename T, typename... Others>
std::ostringstream& _make_msg(std::ostringstream& os, T const& first, Others const&... others)
{
    _make_msg(os, (first));
    return _make_msg(os, (others)...);
}

template <typename... Others>
std::string make_msg(Others const&... others)
{
    std::ostringstream buffer;
    _make_msg(buffer, (others)...);
    return buffer.str();
}
/** @} */

} // namespace logger

/**
 *  @name   Shortcuts for logging
 *  @{
 */
#define SHORT_FILE_LINE_STAMP "[" << (__FILE__) << ":" << (__LINE__) << "] "

#define FILE_LINE_STAMP "From [ " << (__FILE__) << ":" << (__LINE__) << ":0: " << (__PRETTY_FUNCTION__) << " ] "

#define FILE_LINE_STAMP_STRING \
    ("[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ":0: " + std::string(__PRETTY_FUNCTION__) + "] ")
#define MAKE_ERROR_MSG(...) \
    sp::logger::make_msg(" From [", (__FILE__), ":", (__LINE__), ":0: ", (__PRETTY_FUNCTION__), "] \t", __VA_ARGS__)

// sp::logger::make_error_msg( (__FILE__),(__LINE__), (__PRETTY_FUNCTION__),__VA_ARGS__)

#define DONE sp::logger::done

#define WARNING sp::logger::Logger(sp::logger::LOG_WARNING) << FILE_LINE_STAMP
#define TIME_STAMP                              \
    sp::logger::Logger(sp::logger::LOG_VERBOSE) \
        << (__FILE__) << ":" << (__LINE__) << ":0: " << (__PRETTY_FUNCTION__)

#define FUNCTION_START sp::logger::Logger(sp::logger::LOG_VERBOSE) << FILE_LINE_STAMP << " START " << std::endl
#define FUNCTION_END sp::logger::Logger(sp::logger::LOG_VERBOSE) << FILE_LINE_STAMP << " END " << std::endl

#define INFORM sp::logger::Logger(sp::logger::LOG_INFORM)

#define NEED_OPTIMIZATION                       \
    sp::logger::Logger(sp::logger::LOG_VERBOSE) \
        << FILE_LINE_STAMP << "This function should be optimized!" << std::endl
#define UNSUPPORTED                             \
    sp::logger::Logger(sp::logger::LOG_WARNING) \
        << FILE_LINE_STAMP << "UNSUPPORTED!! I won't  do this!" << std::endl
#define UNIMPLEMENTED                           \
    sp::logger::Logger(sp::logger::LOG_WARNING) \
        << FILE_LINE_STAMP << "Sorry, this function is not implemented! " << std::endl

#define TODO                                                                    \
    sp::logger::Logger(sp::logger::LOG_VERBOSE) << FILE_LINE_STAMP << std::endl \
                                                << " \e[32;1m[ TODO  ]\e[0m "

#define FIXME                                                                   \
    sp::logger::Logger(sp::logger::LOG_WARNING) << FILE_LINE_STAMP << std::endl \
                                                << " \e[32;1m[ FIXME ]\e[0m "

#define DUMMY                                                                   \
    sp::logger::Logger(sp::logger::LOG_WARNING) << FILE_LINE_STAMP << std::endl \
                                                << " \e[32;1m[ DUMMY ]\e[0m "
#define DO_NOTHING \
    sp::logger::Logger(sp::logger::LOG_WARNING) << FILE_LINE_STAMP << "NOTHING TO DO" << std::endl

#define OBSOLETE                                                                                              \
    sp::logger::Logger(sp::logger::LOG_WARNING) << FILE_LINE_STAMP << "The function [" << __PRETTY_FUNCTION__ \
                                                << "] is obsolete. Please do not use  it any more."

#define CHANGE_INTERFACE(_MSG_)                                                       \
    sp::logger::Logger(sp::logger::LOG_WARNING)                                       \
        << "[" << __FILE__ << ":" << __LINE__ << ":" << (__PRETTY_FUNCTION__) << "]:" \
        << "The function [" << __PRETTY_FUNCTION__ << "] is obsolete. Please use [" << _MSG_ << "] inside."

#define UNIMPLEMENTED2(_MSG_) THROW_EXCEPTION_RUNTIME_ERROR(_MSG_)

#define UNDEFINE_FUNCTION                                                             \
    sp::logger::Logger(sp::logger::LOG_WARNING)                                       \
        << "[" << __FILE__ << ":" << __LINE__ << ":" << (__PRETTY_FUNCTION__) << "]:" \
        << "This function is not defined!"

#define NOTHING_TODO                                                                  \
    sp::logger::Logger(sp::logger::LOG_VERBOSE)                                       \
        << "[" << __FILE__ << ":" << __LINE__ << ":" << (__PRETTY_FUNCTION__) << "]:" \
        << "oh....... NOTHING TODO!"

#define DEADEND                                                                       \
    sp::logger::Logger(sp::logger::LOG_DEBUG)                                         \
        << "[" << __FILE__ << ":" << __LINE__ << ":" << (__PRETTY_FUNCTION__) << "]:" \
        << "WHAT YOU DO!! YOU SHOULD NOT GET HERE!!"

#define LOGGER sp::logger::Logger(sp::logger::LOG_LOG) << FILE_LINE_STAMP

#define MESSAGE sp::logger::Logger(sp::logger::LOG_MESSAGE)

#define VERBOSE sp::logger::Logger(sp::logger::LOG_VERBOSE) << FILE_LINE_STAMP

#define SHOW_ERROR sp::logger::Logger(sp::logger::LOG_ERROR) << FILE_LINE_STAMP

#define SHOW_WARNING sp::logger::Logger(sp::logger::LOG_WARNING) << FILE_LINE_STAMP

#define RUNTIME_ERROR sp::logger::Logger(sp::logger::LOG_ERROR_RUNTIME) << FILE_LINE_STAMP

#define LOGIC_ERROR sp::logger::Logger(sp::logger::LOG_ERROR_LOGICAL) << FILE_LINE_STAMP

#define BAD_CAST sp::logger::Logger(sp::logger::LOG_ERROR_BAD_CAST) << FILE_LINE_STAMP

#define OUT_OF_RANGE sp::logger::Logger(sp::logger::LOG_ERROR_OUT_OF_RANGE) << FILE_LINE_STAMP

#define INVALID_ARGUMENT sp::logger::Logger(sp::logger::LOG_ERROR_INVALID_ARGUMENT) << FILE_LINE_STAMP

#define DOMAIN_ERROR sp::logger::Logger(sp::logger::LOG_ERROR_DOMAIN) << FILE_LINE_STAMP

#define EXCEPTION_BAD_ALLOC sp::logger::Logger(sp::logger::LOG_ERROR_OUT_OF_RANGE) << FILE_LINE_STAMP
//#define THROW_EXCEPTION(_MSG_) { {logger::Logger(sp::logger::LOG_ERROR)
//<<"["<<__FILE__<<":"<<__LINE__<<":"<<
//(__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::logic_error("error"));}
//
#define THROW_EXCEPTION(_MSG_)               \
    {                                        \
        RUNTIME_ERROR << _MSG_ << std::endl; \
    }
//
////#define THROW_EXCEPTION_RUNTIME_ERROR(_MSG_) { {logger::Logger(sp::logger::LOG_ERROR)
///<<"["<<__FILE__<<":"<<__LINE__<<":"<<
///(__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::runtime_error("runtime error"));}
//
#define THROW_EXCEPTION_RUNTIME_ERROR(...)                        \
    {                                                             \
        auto msg = MAKE_ERROR_MSG(__VA_ARGS__);                   \
        sp::logger::Logger(sp::logger::LOG_ERROR_RUNTIME) << msg; \
        throw std::runtime_error(msg);                            \
    }
//
////#define THROW_EXCEPTION_LOGIC_ERROR(_MSG_)  {{logger::Logger(sp::logger::LOG_ERROR)
///<<"["<<__FILE__<<":"<<__LINE__<<":"<<
///(__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::logic_error("logic error"));}
#define THROW_EXCEPTION_LOGIC_ERROR(_MSG_) \
    {                                      \
        LOGIC_ERROR << _MSG_ << std::endl; \
    }
//
////#define THROW_EXCEPTION_OUT_OF_RANGE(_MSG_) { {logger::Logger(sp::logger::LOG_ERROR)
///<<"["<<__FILE__<<":"<<__LINE__<<":"<<
///(__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::out_of_range("out of entity_id_range"));}
#define THROW_EXCEPTION_OUT_OF_RANGE(...)                        \
    {                                                            \
        auto msg = MAKE_ERROR_MSG("OUT OF RANGE! ",__VA_ARGS__);                  \
        sp::logger::Logger(sp::logger::LOG_ERROR_OUT_OF_RANGE) << msg; \
        throw std::out_of_range(msg);                            \
    }
//
//#define THROW_EXCEPTION_BAD_ALLOC(_SIZE_, _error_)    sp::logger::Logger(sp::logger::LOG_ERROR)<<__FILE__<<"["<<__LINE__<<"]: "<< "Can not Serialize enough memory! [ "  \
//        << _SIZE_ / 1024.0 / 1024.0 / 1024.0 << " GiB ]" << std::endl; throw(_error_);
//
#define THROW_EXCEPTION_BAD_ALLOC(_SIZE_)                                                                             \
    {                                                                                                                 \
        LOGGER << FILE_LINE_STAMP << "Can not get enough memory! [ " << _SIZE_ / 1024.0 / 1024.0 / 1024.0 << " GiB ]" \
               << std::endl;                                                                                          \
        throw(std::bad_alloc());                                                                                      \
    }
//
//

#define THROW_EXCEPTION_BAD_CAST(_FIRST_, _SECOND_)                                          \
    {                                                                                        \
        BAD_CAST << "Can not cast " << (_FIRST_) << " to " << (_SECOND_) << "" << std::endl; \
    }
//

////#define THROW_EXCEPTION_PARSER_ERROR(_MSG_)  {{
/// sp::logger::Logger(sp::logger::LOG_ERROR)<<"["<<__FILE__<<":"<<__LINE__<<":"<<
///(__PRETTY_FUNCTION__)<<"]:"<<"\n\tConfigure fails :"<<(_MSG_) ;}throw(std::runtime_error(""));}
#define THROW_EXCEPTION_PARSER_ERROR(...) throw(std::logic_error(MAKE_ERROR_MSG("Configure fails:", __VA_ARGS__)));

#define PARSER_WARNING(_MSG_)                                                                 \
    {                                                                                         \
        {                                                                                     \
            sp::logger::Logger(sp::logger::LOG_WARNING)                                       \
                << "[" << __FILE__ << ":" << __LINE__ << ":" << (__PRETTY_FUNCTION__) << "]:" \
                << "\n\tConfigure fails :" << (_MSG_);                                        \
        }                                                                                     \
        throw(std::runtime_error(""));                                                        \
    }

#define TRY_IT(_CMD_)                                                                 \
    try                                                                               \
    {                                                                                 \
        _CMD_;                                                                        \
    }                                                                                 \
    catch (std::exception const& _error)                                              \
    {                                                                                 \
        RUNTIME_ERROR << "[" << __STRING(_CMD_) << "]" << _error.what() << std::endl; \
    }

#define TRY_IT1(_CMD_, ...)                                                                       \
    try                                                                                           \
    {                                                                                             \
        _CMD_;                                                                                    \
    }                                                                                             \
    catch (std::exception const& error)                                                           \
    {                                                                                             \
        THROW_EXCEPTION_RUNTIME_ERROR(__VA_ARGS__, ":", "[", __STRING(_CMD_), "]", error.what()); \
    }

//#ifndef NDEBUG
#define CHECK(_MSG_)                                                                                         \
    std::cerr << "From [" << (__FILE__) << ":" << (__LINE__) << ":0: " << (__PRETTY_FUNCTION__) << "] \n \t" \
              << __STRING((_MSG_)) << " = " << std::boolalpha << (_MSG_) << std::endl
#define SHOW(_MSG_) \
    sp::logger::Logger(sp::logger::LOG_VERBOSE) << __STRING(_MSG_) << "\t= " << (_MSG_) << std::endl;
#define SHOW_HEX(_MSG_)                         \
    sp::logger::Logger(sp::logger::LOG_VERBOSE) \
        << __STRING(_MSG_) << "\t= " << std::hex << (_MSG_) << std::dec << std::endl;

//#else
//#	define CHECK(_MSG_)
//#endif

#define REDUCE_CHECK(_MSG_)                                                                     \
    {                                                                                           \
        auto __a = (_MSG_);                                                                     \
        __a = reduce(__a);                                                                      \
        if (GLOBAL_COMM.get_rank() == 0)                                                        \
        {                                                                                       \
            sp::logger::Logger(sp::logger::LOG_DEBUG)                                           \
                << " " << (__FILE__) << ": line " << (__LINE__) << ":" << (__PRETTY_FUNCTION__) \
                << "\n\t GLOBAL_SUM:" << __STRING(_MSG_) << "=" << __a;                         \
        }                                                                                       \
    }

#define RIGHT_COLUMN(_FIRST_) MESSAGE << std::setw(15) << std::right << _FIRST_
#define LEFT_COLUMN(_FIRST_) MESSAGE << std::setw(15) << std::left << _FIRST_

#define INFORM2(_MSG_) sp::logger::Logger(sp::logger::LOG_INFORM) << __STRING(_MSG_) << " = " << _MSG_;

#define DOUBLELINE std::setw(sp::logger::get_line_width()) << std::setfill('=') << "="
#define SINGLELINE std::setw(sp::logger::get_line_width()) << std::setfill('-') << "-"

#define SEPERATOR(_C_) std::setw(sp::logger::get_line_width()) << std::setfill(_C_) << _C_
#define CMD VERBOSE << "CMD:\t"

#define LOG_CMD(_CMD_)                                                           \
    try                                                                          \
    {                                                                            \
        sp::logger::Logger __logger(sp::logger::LOG_VERBOSE);                    \
        __logger << "CMD:\t" << std::string(__STRING(_CMD_));                    \
        _CMD_;                                                                   \
        __logger << DONE;                                                        \
    }                                                                            \
    catch (std::exception const& error)                                          \
    {                                                                            \
        RUNTIME_ERROR << ("[", __STRING(_CMD_), "]", error.what()) << std::endl; \
    }

#define LOG_CMD_DESC(_DESC_, _CMD_)                                              \
    try                                                                          \
    {                                                                            \
        sp::logger::Logger __logger(sp::logger::LOG_VERBOSE);                    \
        __logger << "CMD:\t" << _DESC_;                                          \
        _CMD_;                                                                   \
        __logger << DONE;                                                        \
    }                                                                            \
    catch (std::exception const& error)                                          \
    {                                                                            \
        RUNTIME_ERROR << ("[", __STRING(_CMD_), "]", error.what()) << std::endl; \
    }

#define VERBOSE_CMD(_CMD_)                                    \
    {                                                         \
        sp::logger::Logger __logger(sp::logger::LOG_VERBOSE); \
        __logger << __STRING(_CMD_);                          \
        try                                                   \
        {                                                     \
            _CMD_;                                            \
            __logger << DONE;                                 \
        }                                                     \
        catch (...)                                           \
        {                                                     \
            __logger << sp::logger::failed;                   \
        }                                                     \
    }

#define LOG_CMD1(_LEVEL_, _MSG_, _CMD_)              \
    {                                                \
        auto __logger = sp::logger::Logger(_LEVEL_); \
        __logger << _MSG_;                           \
        _CMD_;                                       \
        __logger << DONE;                            \
    }

#ifdef __CUDA__
#define FE_CMD(_CMD_) _CMD_
#else
#define FE_CMD(_CMD_)                                               \
    _Pragma("STDC_FENV_ACCESS = on")                                \
    {                                                               \
        _CMD_;                                                      \
        if (std::fetestexcept(FE_ALL_EXCEPT) & FE_INVALID)          \
        {                                                           \
            WARNING << " FE_INVALID is raised! "                    \
                    << " [" << __STRING(_CMD_) << "]" << std::endl; \
        }                                                           \
        std::feclearexcept(FE_ALL_EXCEPT);                          \
    }
#endif
//#define LOG_CMD2(_MSG_, _CMD_) {auto
//__logger=logger::Logger(sp::logger::LOG_LOG);__logger<<_MSG_<<__STRING(_CMD_);_CMD_;__logger<<DONE;}

#define CHECK_BIT(_MSG_)                                                                                        \
    std::cout << std::setfill(' ') << std::setw(10) << __STRING(_MSG_) << " = 0b" << sp::logger::ShowBit(_MSG_) \
              << std::endl
#define SHOW_BIT(_MSG_)                                                                                         \
    std::cout << std::setfill(' ') << std::setw(80) << __STRING(_MSG_) << " = 0b" << sp::logger::ShowBit(_MSG_) \
              << std::endl

#define CHECK_HEX(_MSG_)                                                                           \
    std::cout << std::setfill(' ') << std::setw(40) << __STRING(_MSG_) << " = 0x" << std::setw(20) \
              << std::setfill('0') << std::hex << (_MSG_) << std::dec << std::endl

#ifndef NDEBUG
#define SP_CMD(_CMD_)                                                                          \
    {                                                                                          \
        std::feclearexcept(FE_ALL_EXCEPT);                                                     \
        _CMD_;                                                                                 \
        int _error = std::fetestexcept(FE_ALL_EXCEPT);                                         \
        std::feclearexcept(FE_ALL_EXCEPT);                                                     \
        if (_error & FE_INVALID)                                                               \
        {                                                                                      \
            RUNTIME_ERROR << "FE_INVALID is raised! [" << __STRING(_CMD_) << "]" << std::endl; \
        }                                                                                      \
    }
#else
#define SP_CMD(_CMD_) _CMD_
#endif
/** @} */

/** @} defgroup Logging*/
#ifdef NDEBUG
#define ASSERT(_COND_)
#else
#define ASSERT(_COND_)                                                                                              \
    if (!(_COND_))                                                                                                  \
    {                                                                                                               \
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Assertion \"" + __STRING(_COND_) + "\" failed! Abort."); \
    }
#endif
#define TRY_CALL(_CMD_)                                                                            \
    try                                                                                            \
    {                                                                                              \
        _CMD_;                                                                                     \
    }                                                                                              \
    catch (std::exception const& _msg_)                                                            \
    {                                                                                              \
        throw std::runtime_error(_msg_.what() + std::string("\n from:") + FILE_LINE_STAMP_STRING + \
                                 "\"" __STRING(_CMD_) + "\" ");                                    \
    }

class NotImplementedException : public std::logic_error
{
public:
    NotImplementedException(std::string const& prefix = "") : std::logic_error{prefix + " Function is not implemented."} {}
};

#define NOT_IMPLEMENTED                                                              \
    {                                                                                \
        sp::logger::Logger(sp::logger::LOG_ERROR_NOT_IMEPLEMENT) << FILE_LINE_STAMP; \
        throw sp::NotImplementedException(FILE_LINE_STAMP_STRING);                   \
    }

} // namespace sp
#endif /* SP_LOGGER_H_ */
