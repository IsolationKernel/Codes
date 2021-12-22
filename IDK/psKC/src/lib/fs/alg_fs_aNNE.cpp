#include <mass_ml/fs/alg_fs_aNNE.h>

#include <mass_ml/ds/ds_memory.h>
#include <mass_ml/ds/ds_tmp_file.h>
#include <mass_ml/math/math.h>

#include <thread>

namespace mass_ml {

  alg_fs_aNNE_c::alg_fs_aNNE_c(std::size_t sets, std::size_t sample_size) : alg_fs_mass_c(sets, sample_size) {
  }

  void alg_fs_aNNE_c::transform(data_source_c & dst, data_source_c const & src, data_source_c const & model) {
    std::unique_ptr<data_source_c> dists = data_source_c::make_unique<ds_tmp_file_c>(model.rows());
    std::unique_ptr<mass_ml::data_source_c> idx = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(sets_);

//    std::unique_ptr<data_source_c> dists = data_source_c::make_unique<ds_memory_c>(model.rows());
//    std::unique_ptr<mass_ml::data_source_c> idx = mass_ml::data_source_c::make_unique<mass_ml::ds_memory_c>(sets_);

    compute_distance(*dists, src, model);
    find_nearest_neighbour(*dists, *idx, sample_size_);

    std::vector<double> row;
    row.resize(idx->cols(), 0.0);
    std::vector<double> dst_row;
    dst_row.resize(sample_size_ * sets_, 0.0);

    for (std::size_t i = 0; i < src.rows(); i++) {
      std::fill(row.begin(), row.end(), 0.0);
      idx->load_row(row, i);

      std::fill(dst_row.begin(), dst_row.end(), 0.0);

      for (std::size_t j = 0; j < sets_; j++) {
        std::size_t idx = j * sample_size_;

#ifdef NDEBUG
        dst_row[idx + row[j]] = 1.0;
#else
        dst_row.at(idx + row.at(j)) = 1.0;
#endif
      }

      dst.store_row(dst_row, src.label(i));
    }
  }

}
