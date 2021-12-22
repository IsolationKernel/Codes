#include <mass_ml/math/math.h>

#ifndef IS_CUDA

#include <mass_ml/ds/data_source.h>

#include <string>
#include <vector>

#include <cmath>

namespace mass_ml {

  void compute_distance(data_source_c & dist, data_source_c const & a, data_source_c const & b) {
    std::size_t col = a.cols();

    std::vector<double> x;
    x.resize(col, 0.0);
    std::vector<double> y;
    y.resize(col, 0.0);

    for (std::size_t i = 0; i < a.rows(); i++) {
      std::fill(x.begin(), x.end(), 0.0);
      a.load_row(x, i);

      std::vector<double> row_sum;

      for (std::size_t j = 0; j < b.rows(); j++) {
        std::fill(y.begin(), y.end(), 0.0);
        b.load_row(y, j);

        double sum = 0.0;

        for (std::size_t k = 0; k < col; k++) {
#ifdef NDEBUG
          double val = x[k] - y[k];
#else
          double val = x.at(k) - y.at(k);
#endif
          sum += val * val;
        }

        row_sum.push_back(std::sqrt(sum));
      }

      dist.store_row(row_sum, "");
    }
  }

  void find_nearest_neighbour(data_source_c const & dist, data_source_c & idx, std::size_t blk_sz, bool inc_zero) {
    std::size_t entries = dist.cols() / blk_sz;

    std::vector<double> x;
    x.resize(dist.cols(), 0.0);

    for (std::size_t i = 0; i < dist.rows(); i++) {
      std::fill(x.begin(), x.end(), 0.0);
      dist.load_row(x, i);

      std::vector<double> idx_row;

      for (std::size_t j = 0; j < entries; j++) {
        std::size_t offset = blk_sz * j;

        double best_dist = std::numeric_limits<double>::max();
        std::size_t best_idx = 0;

        for (std::size_t k = 0; k < blk_sz; k++) {
  #ifdef NDEBUG
          double val = x[offset + k];
  #else
          double val = x.at(offset + k);
  #endif

          if (!inc_zero && (val == 0.0)) {
            continue;
          }

          if (val < best_dist) {
            best_dist = val;
            best_idx = k;
          }
        }

        idx_row.push_back(best_idx);
      }

      idx.store_row(idx_row, "0");
    }
  }

  void init_math(std::size_t cpu_mem_GiB __attribute__((unused)), std::size_t gpu_mem_GiB __attribute__((unused))) {
  // does nothing for CPU version
  }

}

#endif
