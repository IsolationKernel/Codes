/**
  * The trace code is from the following site:
  *
  * https://github.com/GPMueller/trace
  *
  */

#include <mass_ml/tr/exception.h>

#include <iostream>

namespace trace {

  std::string latest() {
    std::scoped_lock<std::mutex> sl( inner::_trace_string_mutex );
    std::string ret = inner::_latest_trace_string;
    return ret;
  }

}

namespace trace::inner {

  std::string _latest_trace_string = "";
  std::mutex _trace_string_mutex   = std::mutex();

// Backtrace an exception by recursively unwrapping the nested exceptions
  void backtrace( const std::exception & ex ) {
    try {
      std::scoped_lock<std::mutex> sl(_trace_string_mutex);
      _latest_trace_string += std::string(ex.what()) + "\n";
      std::rethrow_if_nested(ex);
    } catch (std::exception const & nested_ex) {
      backtrace(nested_ex);
    }
  }

// TODO
  void throw_exception(std::string const & message, char const * file, unsigned int line) {
    throw exception(message, file, line);
  }

// General Exception handler
  void handle_exception(std::exception const & ex, std::string const & function) {
    try {
      {
        std::scoped_lock<std::mutex> sl(_trace_string_mutex);

        _latest_trace_string = "";

        if (function != "") {
          _latest_trace_string =
            std::string("API: Exception caught in function \'") +
            function + "\'. ";
        }

        _latest_trace_string += "Backtrace:\n";
      }

      backtrace(ex);
    } catch (...) {
      std::cerr << "Something went super-wrong! TERMINATING!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

}
