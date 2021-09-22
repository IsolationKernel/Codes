#ifndef DS_MDL_DEPENDENT_H
#define DS_MDL_DEPENDENT_H

#include <mass_ml/ds/ds_model.h>

#include <random>

namespace mass_ml {

  class ds_mdl_dependent_c : public ds_model_c {
    public:
      virtual std::size_t cols() const override;
      virtual std::size_t rows() const override;

      virtual void load_row(std::vector<double> & row, std::size_t idx) const override;
      virtual std::size_t label_value_count() const override;
      virtual std::string label_value(std::size_t idx) const override;

      virtual std::string label(std::size_t idx) const override;

    protected:
      ds_mdl_dependent_c(uint64_t random_seed, std::size_t sets, std::size_t sample_size, std::shared_ptr<data_source_c> & src);

      virtual void init() override;

    private:
      friend data_source_c;

      uint64_t random_seed_;
      std::size_t sample_size_;
      std::size_t sets_;

      std::shared_ptr<data_source_c> src_;

      mutable std::mt19937_64 mt_random;
      mutable std::uniform_int_distribution<> uid;
      mutable std::vector<std::size_t> sampling_without_replacement;
      mutable std::size_t sample_size_count{0};
      mutable std::vector<std::size_t> idx_to_src;

      std::size_t subsample() const;
      void update_idx(std::size_t idx) const;
  };

  inline std::size_t ds_mdl_dependent_c::cols() const {
    return src_->cols();
  }

  inline std::size_t ds_mdl_dependent_c::label_value_count() const {
    return src_->label_value_count();
  }

  inline std::string ds_mdl_dependent_c::label_value(std::size_t idx) const {
    return src_->label_value(idx);
  }

  inline std::size_t ds_mdl_dependent_c::rows() const {
    return sample_size_ * sets_;
  }

}

#endif // DS_MDL_DEPENDENT_H
