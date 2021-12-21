#ifndef IK_ANNE_H
#define IK_ANNE_H

#include "ik_mass.h"

namespace SOL {

  template <typename FeatType, typename LabelType>
    class ik_aNNE : public ik_mass<FeatType, LabelType> {
      public:
        ik_aNNE(int sets, int psi) : ik_mass<FeatType, LabelType>(sets, psi)  {
        }

        virtual void transform(DataPoint<FeatType, LabelType> const & data, Eigen::VectorXf & zt) override {
          for (int i = 0 ; i < t_sets.size(); i++) {
            std::vector<DataPoint<FeatType, LabelType>> one_set = t_sets[i];
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_idx;

            for (uint32_t j = 0; j < one_set.size(); j++) {
              float dist = distance(one_set[j], data);

              if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
              }
            }

//            zt[i * this->psi_ + best_idx] = 1.0;
            zt[i] = best_idx;
          }
        }

      protected:
        virtual void build_model(std::vector<DataPoint<FeatType, LabelType>> & data_points, std::mt19937 & mt_random) override {
          for (int i = 0; i < this->sets_; i++) {
            std::vector<DataPoint<FeatType, LabelType>> psi_data_points;

            for (int j = 0; j < this->psi_; j++) {
              psi_data_points.push_back(data_points.back());
              data_points.pop_back();
            }

            t_sets.push_back(psi_data_points);
          }
        }

      private:
        std::vector<std::vector<DataPoint<FeatType, LabelType>>> t_sets;

        float distance(DataPoint<FeatType, LabelType> const & x, DataPoint<FeatType, LabelType> const & y) {
          float dist = 0.0f;

          std::size_t idx_x = 0;
          std::size_t idx_y = 0;

          std::size_t count_x = x.indexes.size();
          std::size_t count_y = y.indexes.size();

          while ((idx_x != count_x) && (idx_y != count_y)) {
            if (x.indexes[idx_x] > y.indexes[idx_y]) {
              dist += y.features[idx_y] * y.features[idx_y];
              idx_y++;
            } else if (x.indexes[idx_x] < y.indexes[idx_y]) {
              dist += x.features[idx_x] * x.features[idx_x];
              idx_x++;
            } else {
              float val = x.features[idx_x] - y.features[idx_y];
              dist += val * val;
              idx_x++;
              idx_y++;
            }
          }

          if (idx_x == count_x) {
            for (int i = idx_y; i < count_y; i++) {
              dist += y.features[i] * y.features[i];
            }
          }

          if (idx_y == count_y) {
            for (int i = idx_x; i < count_x; i++) {
              dist += x.features[i] * x.features[i];
            }
          }

//          return std::sqrt(dist);
          return dist;
        }
    };

}

#endif // IK_ANNE_H
