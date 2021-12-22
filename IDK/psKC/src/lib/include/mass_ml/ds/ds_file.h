#ifndef DS_FILE_H
#define DS_FILE_H

#include <mass_ml/ds/data_source.h>

#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace mass_ml {

  enum class ds_file_mode_e : uint8_t {
    read, write
  };

  class ds_file_c : public data_source_c {
    public:
      virtual std::size_t cols() const override;
      virtual std::size_t rows() const override;

      virtual std::size_t label_value_count() const override;
      virtual std::string label_value(std::size_t idx) const override;

    protected:
      std::string const dir_name_;
      std::string const file_name_;

      mutable std::fstream file;

      std::size_t cols_{0};
      std::vector<std::size_t> file_row_idx;

      std::set<std::string> label_values_;

      ds_file_c(std::string const & dir_name, std::string const & file_name, ds_file_mode_e mode);
  };

  inline std::size_t ds_file_c::cols() const {
    return cols_;
  }

  inline std::size_t ds_file_c::label_value_count() const {
    return label_values_.size();
  }

  inline std::string ds_file_c::label_value(std::size_t idx) const {
    return *std::next(label_values_.begin(), idx);
  }

  inline std::size_t ds_file_c::rows() const {
    return file_row_idx.size() - 1;
  }

}

#endif // DS_FILE_H
