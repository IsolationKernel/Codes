#ifndef IK_DOT_PRODUCT_H
#define IK_DOT_PRODUCT_H

#include "ik_mass.h"

namespace SOL {

  template <typename FeatType, typename LabelType>
    class ik_dot_product : public ik_mass<FeatType, LabelType> {
      public:
        ik_dot_product(int sets, int psi) : ik_mass<FeatType, LabelType>(sets, psi) {
        }

        virtual void transform(DataPoint<FeatType, LabelType> const & data, Eigen::VectorXf & zt) override {
//          for (std::size_t i = 0; i < data.indexes.size(); i++) {
//            zt[data.indexes[i] - 1] = data.features[i];
//          }

            for (std::size_t i = 0; i < this->sets_; i++) {
              std::size_t idx = i * this->psi_;
              zt[i] = (data.indexes[i] - 1) - idx;
            }
        }

      protected:
        virtual void build_model(std::vector<DataPoint<FeatType, LabelType>> & data_points, std::mt19937 & mt_random) override {
        // nothing to do here!
        }
    };

}

#endif // IK_DOT_PRODUCT_H
