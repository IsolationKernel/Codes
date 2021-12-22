#ifndef TRACE_EXCEPTION_H
#define TRACE_EXCEPTION_H

/**
  * The trace code is from the following site:
  *
  * https://github.com/GPMueller/trace
  *
  */

#include <mutex>
#include <string>

namespace trace {

// Retrieve the latest (error) message
  std::string latest();

}

namespace trace::inner {

  extern std::string _latest_trace_string;
  extern std::mutex _trace_string_mutex;

// Throw a trace::inner::exception
  void throw_exception(std::string const & message, char const * file, unsigned int line);

// Rethrow (creates a std::nested_exception) an exception, using the Exception class
// which contains file and line info. The original exception is preserved...
  void rethrow_exception(std::string const & message, char const * file, unsigned int line);

// General exception handler
  void handle_exception(std::exception const & ex, std::string const & function = "");

// Custom exception class to be used for more practical throwing
  class exception : public std::runtime_error {
    public:
      exception(std::string const & message, char const * file, unsigned int line) : std::runtime_error(message) {
        _message = std::string(file) + ":" + std::to_string(line) + " : " + message;
      }

      ~exception() throw() = default;

      char const * what() const throw() {
        return _message.c_str();
      }

    private:
      std::string _message;
  };

}

#endif // TRACE_EXCEPTION_H
