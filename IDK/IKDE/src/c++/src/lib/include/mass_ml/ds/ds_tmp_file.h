#ifndef DS_TMP_FILE_H
#define DS_TMP_FILE_H

#include <mass_ml/ds/data_source.h>

#include <set>

namespace mass_ml {

  class ds_tmp_file_c : public data_source_c {
    public:
      virtual ~ds_tmp_file_c();

      virtual std::size_t cols() const override;
      virtual std::size_t rows() const override;

      virtual void load_row(std::vector<double> & row, std::size_t idx) const override;
      virtual void store_row(std::vector<double> const & row, std::string const & label) override;

      virtual std::size_t label_value_count() const override;
      virtual std::string label_value(std::size_t idx) const override;
      virtual std::string label(std::size_t idx) const override;

    protected:
      ds_tmp_file_c(std::size_t col);

      virtual void init() override;

    private:
      friend data_source_c;

      std::size_t col_;

      std::vector<std::string> labels;
      std::set<std::string> label_values_;

      std::vector<double> values;

      mutable std::FILE * temp_file{nullptr};
      mutable char * file_name{nullptr};
  };

  inline std::size_t ds_tmp_file_c::cols() const {
    return col_;
  }

  inline std::size_t ds_tmp_file_c::label_value_count() const {
    return label_values_.size();
  }

  inline std::string ds_tmp_file_c::label_value(std::size_t idx) const {
    return *std::next(label_values_.begin(), idx);
  }

  extern void clean_up_tmp_dir();
  extern void init_tmp_dir(std::string const & tmp_dir);

}

#endif // DS_TMP_FILE_H
