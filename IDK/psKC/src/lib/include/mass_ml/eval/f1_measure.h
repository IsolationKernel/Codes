#ifndef F1_MEASURE_H
#define F1_MEASURE_H

#include <ostream>
#include <vector>

namespace mass_ml {

  class data_source_c;

  void f1_measure(std::ostream & out, data_source_c const & data, std::vector<std::vector<std::size_t>> const & clusters, std::vector<std::size_t> const & noise, bool exit_early = false);

  }

#endif // F1_MEASURE_H
