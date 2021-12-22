#ifndef ALG_FEATURE_SPACE_H
#define ALG_FEATURE_SPACE_H

namespace mass_ml {

  class data_source_c;

  class alg_feature_space_c {
    public:
      virtual ~alg_feature_space_c() = default;

      virtual void transform(data_source_c & dst, data_source_c const & src, data_source_c const & model) = 0;
  };

}

#endif // ALG_FEATURE_SPACE_H
