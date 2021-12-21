#include <mass_ml/ds/ds_mdl_dependent.h>

#include <algorithm>

namespace mass_ml {

  ds_mdl_dependent_c::ds_mdl_dependent_c(uint64_t random_seed, std::size_t sets, std::size_t sample_size, std::shared_ptr<data_source_c> & src) : random_seed_(random_seed), sample_size_(sample_size), sets_(sets), src_(src) {
  }

  void ds_mdl_dependent_c::init() {
    mt_random.seed(random_seed_);
    uid = std::uniform_int_distribution<>(0, src_->rows() - 1);
  }

  std::string ds_mdl_dependent_c::label(std::size_t idx) const {
    update_idx(idx);

#ifdef NDEBUG
    return src_->label(idx_to_src[idx]);
#else
    return src_->label(idx_to_src.at(idx));
#endif
  }

  void ds_mdl_dependent_c::load_row(std::vector<double> & row, std::size_t idx) const {
    update_idx(idx);

#ifdef NDEBUG
    src_->load_row(row, idx_to_src[idx]);
#else
    src_->load_row(row, idx_to_src.at(idx));
#endif
  }

  std::size_t ds_mdl_dependent_c::subsample() const {
    if (sampling_without_replacement.size() == src_->rows()) {
        sampling_without_replacement.erase(sampling_without_replacement.begin(), sampling_without_replacement.end() - sample_size_count);
      }

      std::size_t random_index = uid(mt_random);

      while (std::find(sampling_without_replacement.begin(), sampling_without_replacement.end(), random_index) != sampling_without_replacement.end()) {
        random_index = uid(mt_random);
      }

      sample_size_count = (sample_size_count + 1) % sample_size_;

      return random_index;
  }

  void ds_mdl_dependent_c::update_idx(std::size_t idx) const {
    if ((idx + 1) >= idx_to_src.size()) {
      for (std::size_t i = idx_to_src.size(); i < (idx + 1); i++) {
        idx_to_src.push_back(subsample());
      }
    }
  }

}
