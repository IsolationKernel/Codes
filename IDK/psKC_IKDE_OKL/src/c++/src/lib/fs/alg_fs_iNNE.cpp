#include <mass_ml/fs/alg_fs_iNNE.h>

#include <mass_ml/ds/ds_model.h>
#include <mass_ml/ds/ds_tmp_file.h>
#include <mass_ml/math/math.h>

namespace mass_ml {

  alg_fs_iNNE_c::alg_fs_iNNE_c(std::size_t sets, std::size_t sample_size) : alg_fs_mass_c(sets, sample_size) {
  }

  void alg_fs_iNNE_c::compute_ball(data_source_c & ball, data_source_c const & model) {
    std::vector<double> row;
    std::vector<double> idx_row;
    std::vector<double> dist_row;
    std::vector<double> ball_row;

    row.resize(model.cols(), 0.0);
    idx_row.resize(1, 0.0);
    dist_row.resize(sample_size_, 0.0);
    ball_row.resize(2, 0.0);

    for (std::size_t i = 0; i < sets_; i++) {
      std::unique_ptr<data_source_c> a = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(model.cols());
      std::unique_ptr<data_source_c> b = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(model.cols());

      for (std::size_t j = 0; j < sample_size_; j++) {
        std::fill(row.begin(), row.end(), 0.0);
        model.load_row(row, i * sample_size_ + j);

        a->store_row(row, "0");
        b->store_row(row, "0");
      }

      std::unique_ptr<data_source_c> dists = data_source_c::make_unique<ds_tmp_file_c>(sample_size_);
      compute_distance(*dists, *a, *b);

      std::unique_ptr<data_source_c> idx = data_source_c::make_unique<ds_tmp_file_c>(1);
      find_nearest_neighbour(*dists, *idx, sample_size_, false);

      for (std::size_t j = 0; j < sample_size_; j++) {
        std::fill(idx_row.begin(), idx_row.end(), 0.0);
        idx->load_row(idx_row, j);

        std::fill(dist_row.begin(), dist_row.end(), 0.0);
        dists->load_row(dist_row, j);

        ball_row[0] = idx_row[0];
        ball_row[1] = dist_row[idx_row[0]];
        ball.store_row(ball_row, "0");
      }
    }
  }

  void alg_fs_iNNE_c::transform(data_source_c & dst, data_source_c const & src, data_source_c const & model) {
    std::unique_ptr<data_source_c> ball = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(2);
    compute_ball(*ball, model);

    std::unique_ptr<data_source_c> dists = data_source_c::make_unique<ds_tmp_file_c>(model.rows());
    std::unique_ptr<data_source_c> idx = data_source_c::make_unique<ds_tmp_file_c>(sets_);

    compute_distance(*dists, src, model);
    find_nearest_neighbour(*dists, *idx, sample_size_);
\
    std::vector<double> dist_row;
    dist_row.resize(model.rows(), 0.0);
    std::vector<double> row;
    row.resize(idx->cols(), 0.0);
    std::vector<double> dst_row;
    dst_row.resize(sample_size_ * sets_, 0.0);
    std::vector<double> ball_row;
    ball_row.resize(2, 0.0);

    for (std::size_t i = 0; i < src.rows(); i++) {
      std::fill(row.begin(), row.end(), 0.0);
      idx->load_row(row, i);
      std::fill(dist_row.begin(), dist_row.end(), 0.0);
      dists->load_row(dist_row, i);

      std::fill(dst_row.begin(), dst_row.end(), 0.0);

      for (std::size_t j = 0; j < sets_; j++) {
        std::size_t idx = j * sample_size_;

  #ifdef NDEBUG
        ball->load_row(ball_row, idx + row[j]);

        if (dist_row[idx + row[j]] < ball_row[1]) {
          dst_row[idx + row[j]] = 1.0;
        }
  #else
        ball->load_row(ball_row, idx + row.at(j));

        if (dist_row.at(idx + row.at(j)) < ball_row.at(1)) {
          dst_row.at(idx + row.at(j)) = 1.0;
        }
  #endif
      }

      dst.store_row(dst_row, src.label(i));
    }
  }

}
