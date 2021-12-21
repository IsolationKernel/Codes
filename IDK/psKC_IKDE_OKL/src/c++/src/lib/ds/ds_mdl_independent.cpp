#include <mass_ml/ds/ds_mdl_independent.h>

namespace mass_ml {

  ds_mdl_independent_c::ds_mdl_independent_c(uint64_t random_seed, std::size_t rows, std::size_t cols) : random_seed_(random_seed), rows_(rows), cols_(cols) {
  }

  void ds_mdl_independent_c::init() {
    mt_random.seed(random_seed_);
    udd = std::uniform_real_distribution<>(0.0, 1.0);
  }

  void ds_mdl_independent_c::load_row(std::vector<double> & row, std::size_t idx) const {
    if (idx == 0) {
      mt_random.seed(random_seed_);
    }

    for (std::size_t i = 0; i < cols_; i++) {
#ifdef NDEBUG
      row[i] = udd(mt_random);
#else
      row.at(i) = udd(mt_random);
#endif
    }
  }

}
