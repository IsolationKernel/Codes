#ifndef ALG_FS_MASS_H
#define ALG_FS_MASS_H

#include <mass_ml/fs/alg_feature_space.h>

#include <cstdint>

namespace mass_ml {

  class alg_fs_mass_c : public alg_feature_space_c {
    public:
      alg_fs_mass_c(std::size_t sets, std::size_t sample_size);

    protected:
      std::size_t sample_size_;
      std::size_t sets_;
  };

}

#endif // ALG_FS_MASS_H
