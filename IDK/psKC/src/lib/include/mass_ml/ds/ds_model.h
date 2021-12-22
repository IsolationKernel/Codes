#ifndef DS_MODEL_H
#define DS_MODEL_H

#include <mass_ml/ds/data_source.h>
#include <mass_ml/tr/trace.h>

namespace mass_ml {

  class ds_model_c : public data_source_c {
    public:
      virtual void store_row(std::vector<double> const & row, std::string const & label) override;
  };

  inline void ds_model_c::store_row(std::vector<double> const & row __attribute__((unused)), std::string const & label __attribute__((unused))) {
  // does nothing!
    ml_tr_initiate("ds_model_c::store_row - incorrectly called");
  }

}

#endif // DS_MODEL_H
