#ifndef IK_BASE_H
#define IK_BASE_H

#include "../data/DataSet.h"

#include "../Params.h"

#include <Eigen>

namespace SOL {

  template <typename FeatType, typename LabelType>
    class ik_base {
      public:
        virtual ~ik_base() = default;

        virtual void build_model(Params const & param, DataSet<FeatType, LabelType> & dataset) = 0;

        virtual void transform(DataPoint<FeatType, LabelType> const & data, Eigen::VectorXf & zt) = 0;
    };

}

#endif // IK_BASE_H
