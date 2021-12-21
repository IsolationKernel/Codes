#ifndef DS_MEMORY_H
#define DS_MEMORY_H

#include <mass_ml/ds/data_source.h>

#include <set>

namespace mass_ml {

  class ds_memory_c : public data_source_c {
   public:
      virtual std::size_t cols() const override;
      virtual std::size_t rows() const override;

      virtual void load_row(std::vector<double> & row, std::size_t idx) const override;
      virtual void store_row(std::vector<double> const & row, std::string const & label) override;

      virtual std::string label(std::size_t idx) const override;
      virtual std::size_t label_value_count() const override;
      virtual std::string label_value(std::size_t idx) const override;

    protected:
      ds_memory_c(std::size_t col);

      virtual void init() override;

    private:
      friend data_source_c;

      std::size_t cols_;

      std::vector<std::size_t> col_idx;
      std::vector<std::size_t> row_idx;

      std::vector<std::string> labels;
      std::set<std::string> label_values_;

      std::vector<double> values;
  };

  inline std::size_t ds_memory_c::cols() const {
    return cols_;
  }

  inline std::size_t ds_memory_c::label_value_count() const {
    return label_values_.size();
  }

  inline std::string ds_memory_c::label_value(std::size_t idx) const {
    return *std::next(label_values_.begin(), idx);
  }

  inline std::size_t ds_memory_c::rows() const {
    return row_idx.size() - 1;
  }

}

#endif // DS_MEMORY_H
