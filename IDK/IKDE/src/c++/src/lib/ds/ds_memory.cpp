#include <mass_ml/ds/ds_memory.h>

#include <mass_ml/tr/trace.h>

namespace mass_ml {

  ds_memory_c::ds_memory_c(std::size_t col) : cols_(col) {
  }

  void ds_memory_c::init() {
    row_idx.push_back(0);
  }

  std::string ds_memory_c::label(std::size_t idx) const {
#ifdef NDEBUG
    return labels[idx];
#else
    return labels.at(idx);
#endif
  }

  void ds_memory_c::load_row(std::vector<double> & row, std::size_t idx) const {
#ifdef NDEBUG
    std::size_t start = row_idx[idx];
    std::size_t end = row_idx[idx + 1];
#else
    std::size_t start = row_idx.at(idx);
    std::size_t end = row_idx.at(idx + 1);
#endif

    for (std::size_t idx = start; idx < end; idx++) {
#ifdef NDEBUG
      row[col_idx[idx]] = values[idx];
#else
      row.at(col_idx.at(idx)) = values.at(idx);
#endif
    }
  }

  void ds_memory_c::store_row(std::vector<double> const & row, std::string const & label) {
    labels.push_back(label);
    label_values_.insert(label);

    for (std::size_t i = 0; i < row.size(); i++) {
#ifdef NDEBUG
      double val = row[i];
#else
      double val = row.at(i);
#endif

      if (val != 0.0) {
        col_idx.push_back(i);
        values.push_back(val);
      }
    }

    row_idx.push_back(col_idx.size());
  }

}
