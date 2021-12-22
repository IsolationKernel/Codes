#include <mass_ml/math/math.h>

#include <mass_ml/ds/ds_tmp_file.h>

#include <iostream>
#include <limits>

// CUDA Runtime
//#include <cuda.h>
//#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>
//#include <helper_functions.h>

/*
 * The following CUDA kernel codes are from the CUDA samples with modification.
 *
 * - compute_dist_p_2 : 0_Simple/matrixMUL
 *
 */


constexpr uint16_t BLOCK_SIZE = 32;

namespace gpu_matrix_dense {

  __global__
  void compute_dist_p_2(double * C, double * A, double * B, int wA, int wB) {
  // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

  // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
    int bBegin = wA * BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
    double Csub = 0.0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
      __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
      __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
      As[ty][tx] = A[a + wA * ty + tx];
      Bs[tx][ty] = B[b + wA * ty + tx];

    // Synchronize to make sure the matrices are loaded
      __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        double val = As[ty][k] - Bs[k][tx];
        Csub += val * val;
//        Csub += std::pow(val, 2.0);
      }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] += Csub;
  }

  __global__
  void do_sqrt(double * C, int wB) {
  // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

  // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = std::sqrt(C[c + wB * ty + tx]);
  }

  __global__
  void compute_NN(double * C, double * A, int wA, int wB) {
  }

}

namespace {

  struct stats_s {
    std::size_t cols;
    std::size_t a_rows;
    std::size_t b_rows;

    std::size_t act_cols;
    std::size_t act_a_rows;
    std::size_t act_b_rows;

    std::size_t blk_sz_cols;
    std::size_t blk_sz_a_rows;
    std::size_t blk_sz_b_rows;

    std::size_t blk_cnt_cols;
    std::size_t blk_cnt_a_rows;
    std::size_t blk_cnt_b_rows;
  };

  std::size_t max_cpu_mem_GiB{4ull * 1024 * 1024 * 1024};
  std::size_t max_gpu_mem_GiB{4ull * 1024 * 1024 * 1024};

  inline std::size_t compute_minimum_entries(std::size_t val) {
    return val + ((((val / BLOCK_SIZE) * BLOCK_SIZE) == val) ? 0 : BLOCK_SIZE - (val % BLOCK_SIZE));;
  }

  void compute_dist_stats(stats_s & stat) {
    stat.act_cols = compute_minimum_entries(stat.cols);
    stat.act_a_rows = compute_minimum_entries(stat.a_rows);
    stat.act_b_rows = compute_minimum_entries(stat.b_rows);

    stat.blk_sz_cols = BLOCK_SIZE;
    stat.blk_sz_a_rows = BLOCK_SIZE;
    stat.blk_sz_b_rows = BLOCK_SIZE;

    for (std::size_t i = 1; (i <= stat.act_cols) || (i <= stat.act_a_rows) || (i <= stat.act_b_rows); i++) {
      std::size_t blk_sz = i * BLOCK_SIZE;
      std::size_t temp_cols = (blk_sz < stat.act_cols) ? blk_sz : stat.act_cols;
      std::size_t temp_a_rows = (blk_sz < stat.act_a_rows) ? blk_sz : stat.act_a_rows;
      std::size_t temp_b_rows = (blk_sz < stat.act_b_rows) ? blk_sz : stat.act_b_rows;

      if ((8ull * ((temp_a_rows * temp_b_rows) + (temp_cols * temp_a_rows) + (temp_cols * temp_b_rows))) > max_gpu_mem_GiB) {
        break;
      }

      stat.blk_sz_cols = temp_cols;
      stat.blk_sz_a_rows = temp_a_rows;
      stat.blk_sz_b_rows = temp_b_rows;
    }

    stat.blk_cnt_cols = stat.act_cols / stat.blk_sz_cols;
    stat.blk_cnt_a_rows = stat.act_a_rows / stat.blk_sz_a_rows;
    stat.blk_cnt_b_rows = stat.act_b_rows / stat.blk_sz_b_rows;

    if ((stat.blk_cnt_cols * stat.blk_sz_cols) < stat.act_cols) {
      stat.blk_cnt_cols++;
    }

    if ((stat.blk_cnt_a_rows * stat.blk_sz_a_rows) < stat.act_a_rows) {
      stat.blk_cnt_a_rows++;
    }

    if ((stat.blk_cnt_b_rows * stat.blk_sz_b_rows) < stat.act_b_rows) {
      stat.blk_cnt_b_rows++;
    }
  }

  void load_data(double * dst, mass_ml::data_source_c const & src, std::size_t src_idx, std::size_t src_blk_sz, std::size_t col_idx, std::size_t col_blk_sz) {
    checkCudaErrors(cudaMemset(dst, 0x00, sizeof(double) * src_blk_sz * col_blk_sz));

    std::vector<double> vals;
    vals.resize(src.cols(), 0.0);

    std::size_t src_offset = src_idx * src_blk_sz;
    std::size_t col_offset = col_idx * col_blk_sz;

    for (std::size_t i = 0; i < src_blk_sz; i++) {
      if ((src_offset + i) < src.rows()) {
        std::fill(vals.begin(), vals.end(), 0.0);
        src.load_row(vals, src_offset + i);

        for (std::size_t j = 0; j < col_blk_sz; j++) {
          if ((col_offset + j) < src.cols()) {
            dst[i * col_blk_sz + j] = vals[col_offset + j];
          }
        }
      }
    }
  }

  void load_data(double * dst, mass_ml::data_source_c const & src, std::size_t row_idx, std::size_t row_blk_sz, std::size_t col_idx, std::size_t col_blk_sz, std::size_t cols) {

  }

  void store_results(mass_ml::data_source_c & results, double * dist, std::size_t sz_a, std::size_t sz_b) {
    for (std::size_t i = 0; i < sz_a; i++) {
      std::vector<double> row;

      for (std::size_t j = 0; j < sz_b; j++) {
        row.push_back(dist[i * sz_b + j]);
      }

      results.store_row(row, "0");
    }
  }

  void store_results(mass_ml::data_source_c & dist, mass_ml::data_source_c const & results, std::size_t sz_a, std::size_t sz_b, std::size_t idx_a, std::size_t max_a, std::size_t max_b) {
    std::size_t blk_cnt = results.rows() / sz_a;

    std::size_t a_offset = idx_a * sz_a;

    std::vector<double> res_row;
    res_row.resize(sz_b, 0.0);

    for (std::size_t i = 0; i < sz_a; i++) {
      if ((a_offset + i) < max_a) {
        std::vector<double> row;

        for (std::size_t j = 0; j < blk_cnt; j++) {
          std::fill(res_row.begin(), res_row.end(), 0.0);
          results.load_row(res_row, j * sz_a + i);

          for (std::size_t k = 0; k < sz_b; k++) {
            if ((j * sz_b + k) < max_b) {
              row.push_back(res_row[k]);
            }
          }
        }

        dist.store_row(row, "");
      }
    }
  }

}

namespace mass_ml {


  void compute_distance(data_source_c & dist, data_source_c const & a, data_source_c const & b) {
    stats_s stat;
    stat.cols = a.cols();
    stat.a_rows = a.rows();
    stat.b_rows = b.rows();

    compute_dist_stats(stat);

    std::cout << "gpu::compute_distance:" << std::endl;
    std::cout << "- cols    : " << stat.cols << " (" << stat.act_cols << ") - " << stat.blk_sz_cols << " (" << stat.blk_cnt_cols << ")" << std::endl;;
    std::cout << "- rows (a): " << stat.a_rows << " (" << stat.act_a_rows << ") - " << stat.blk_sz_a_rows << " (" << stat.blk_cnt_a_rows << ")" << std::endl;;
    std::cout << "- rows (b): " << stat.b_rows << " (" << stat.act_b_rows << ") - " << stat.blk_sz_b_rows << " (" << stat.blk_cnt_b_rows << ")" << std::endl;;

    double * temp_dists;
    double * temp_a_rows;
    double * temp_b_rows;
    checkCudaErrors(cudaMallocManaged(&temp_dists, sizeof(double) * stat.blk_sz_a_rows * stat.blk_sz_b_rows));
    checkCudaErrors(cudaMallocManaged(&temp_a_rows, sizeof(double) * stat.blk_sz_a_rows * stat.blk_sz_cols));
    checkCudaErrors(cudaMallocManaged(&temp_b_rows, sizeof(double) * stat.blk_sz_b_rows * stat.blk_sz_cols));

    for (std::size_t i = 0; i < stat.blk_cnt_a_rows; i++) {
      std::unique_ptr<data_source_c> results = data_source_c::make_unique<ds_tmp_file_c>(stat.blk_sz_b_rows);

      for (std::size_t j = 0; j < stat.blk_cnt_b_rows; j++) {
        checkCudaErrors(cudaMemset(temp_dists, 0x00, sizeof(double) * stat.blk_sz_a_rows * stat.blk_sz_b_rows));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(stat.blk_sz_b_rows / block.x, stat.blk_sz_a_rows / block.y);

        for (std::size_t k = 0; k < stat.blk_cnt_cols; k++) {
          load_data(temp_a_rows, a, i, stat.blk_sz_a_rows, k, stat.blk_sz_cols);
          load_data(temp_b_rows, b, j, stat.blk_sz_b_rows, k, stat.blk_sz_cols);

          gpu_matrix_dense::compute_dist_p_2<<<grid, block>>>(temp_dists, temp_a_rows, temp_b_rows, stat.blk_sz_cols, stat.blk_sz_b_rows);
          getLastCudaError("Kernel execution failed: gpu_matrix_dense::compute_distance");
          cudaDeviceSynchronize();
        }

        gpu_matrix_dense::do_sqrt<<<grid, block>>>(temp_dists, stat.blk_sz_b_rows);
        getLastCudaError("Kernel execution failed: gpu_matrix_dense::do_sqrt");
        cudaDeviceSynchronize();

        store_results(*results, temp_dists, stat.blk_sz_a_rows, stat.blk_sz_b_rows);
      }

      store_results(dist, *results, stat.blk_sz_a_rows, stat.blk_sz_b_rows, i, a.rows(), b.rows());
    }

    cudaFree(temp_a_rows);
    cudaFree(temp_b_rows);
    cudaFree(temp_dists);
  }

  void init_math(std::size_t cpu_mem_GiB, std::size_t gpu_mem_GiB) {
    max_cpu_mem_GiB = cpu_mem_GiB;
    max_gpu_mem_GiB = gpu_mem_GiB;
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

}
