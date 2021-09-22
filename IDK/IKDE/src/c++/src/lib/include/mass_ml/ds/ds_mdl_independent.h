#ifndef DS_MDL_INDEPENDENT_H
#define DS_MDL_INDEPENDENT_H

#include <mass_ml/ds/ds_model.h>

#include <random>

namespace mass_ml {

  class ds_mdl_independent_c : public ds_model_c {
    public:
      virtual std::size_t cols() const override;
      virtual std::size_t rows() const override;

      virtual void load_row(std::vector<double> & row, std::size_t idx) const override;
      virtual std::size_t label_value_count() const override;
      virtual std::string label_value(std::size_t idx) const override;

      virtual std::string label(std::size_t idx) const override;

    protected:
      ds_mdl_independent_c(uint64_t random_seed, std::size_t rows, std::size_t cols);

      virtual void init() override;

    private:
      friend data_source_c;

      uint64_t random_seed_;
      std::size_t rows_;
      std::size_t cols_;

      mutable std::mt19937_64 mt_random;
      mutable std::uniform_real_distribution<> udd;
  };

  inline std::size_t ds_mdl_independent_c::cols() const {
    return cols_;
  }

  inline std::string ds_mdl_independent_c::label(std::size_t idx) const {
  // does nothing!
    ml_tr_initiate("ds_mdl_independent_c::label - incorrectly called");
  }

  inline std::size_t ds_mdl_independent_c::label_value_count() const {
  // does nothing!
    ml_tr_initiate("ds_mdl_independent_c::label_value_count - incorrectly called");
  }

  inline std::string ds_mdl_independent_c::label_value(std::size_t idx __attribute__((unused))) const {
  // does nothing!
    ml_tr_initiate("ds_mdl_independent_c::label_value - incorrectly called");
  }

  inline std::size_t ds_mdl_independent_c::rows() const {
    return rows_;
  }

}

#endif // DS_MDL_INDEPENDENT_H
