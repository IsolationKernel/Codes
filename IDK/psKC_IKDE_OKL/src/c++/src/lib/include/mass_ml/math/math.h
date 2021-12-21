#ifndef MATH_H
#define MATH_H

#include <vector>

namespace mass_ml {

  class data_source_c;

  extern void compute_distance(data_source_c & dist, data_source_c const & a, data_source_c const & b);

  extern void find_nearest_neighbour(data_source_c const & dist, data_source_c & idx, std::size_t blk_sz, bool inc_zero = true);

  extern void init_math(std::size_t cpu_mem_GiB, std::size_t gpu_mem_GiB);
}

#endif // MATH_H
