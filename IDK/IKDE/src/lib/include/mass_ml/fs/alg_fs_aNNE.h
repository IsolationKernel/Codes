#ifndef ALG_FS_ANNE_H
#define ALG_FS_ANNE_H

#include <mass_ml/fs/alg_fs_mass.h>

namespace mass_ml {

  class alg_fs_aNNE_c : public alg_fs_mass_c {
    public:
      alg_fs_aNNE_c(std::size_t sets, std::size_t sample_size);

      virtual void transform(data_source_c & dst, data_source_c const & src, data_source_c const & model) override;
  };

}

#endif // ALG_FS_ANNE_H
