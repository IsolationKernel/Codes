#ifndef PARSE_PARAMETER_H
#define PARSE_PARAMETER_H

#include <mass_ml/tr/trace.h>

#include <algorithm>
#include <sstream>

namespace mass_ml {

// https://stackoverflow.com/questions/865668/how-to-parse-command-line-arguments-in-c
  template <typename T>
    void parse_parameter(int argc, char ** argv, std::string const & option, T & result, std::string const & err_msg, bool abort = true) {
      char ** end = argv + argc;
      char ** itr = std::find(argv, end, option);

      if ((itr != end) && (++itr != end)) {
        std::stringstream is(*itr);
        is >> std::skipws;
        is >> result;
      } else {
        if (abort) {
          ml_tr_initiate(err_msg);
        }
      }
    }

  void parse_parameter(int argc, char ** argv, std::string const & option, bool & result) {
    char ** end = argv + argc;
    char ** itr = std::find(argv, end, option);

    result = (itr != end);
  }

}

#endif // PARSE_PARAMETER_H
