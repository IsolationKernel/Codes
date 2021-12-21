#ifndef IK_IFOREST_H
#define IK_IFOREST_H

#include "ik_mass.h"

#include <unordered_set>

namespace SOL {

  template <typename FeatType, typename LabelType>
    class node {
      public:
        void build_tree(std::vector<DataPoint<FeatType, LabelType>> data, int n_attr, std::mt19937 & random, int current_height, int max_height, uint32_t min_pts, int & ID) {
          if ((current_height >= max_height) || (data.size() <= min_pts)) {
            leafID = ID++;
            return;
          }

          std::uniform_int_distribution<> uid = std::uniform_int_distribution<>(0, n_attr - 1);
          std::uniform_real_distribution<> udd = std::uniform_real_distribution<>(0, 1);

          std::unordered_set<int> check;
          bool found = false;

          while (check.size() < std::size_t(n_attr)) {
            idx_attr = uid(random);

            if (check.find(idx_attr) != check.end()) {
              continue;
            }

            check.insert(idx_attr);

            double max = get_value(data[0], idx_attr);
            double min = max;

            for (std::size_t i = data.size() - 1; i > 0; i--) {
              double val = get_value(data[i], idx_attr);

              max = std::max(max, val);
              min = std::min(min, val);
            }

            if (max > min) {
              split_point = udd(random) * (max - min) + min;
              found = true;
              break;
            }
          }

          if (!found) {
            leafID = ID++;
            std::cout << "data size: " << data.size() << " - all attributes are equal value!" << std::endl;
            return; // all remaining points are at the same location;
          }

          std::vector<DataPoint<FeatType, LabelType>> left;
          std::vector<DataPoint<FeatType, LabelType>> right;

          for (int i = data.size() - 1; i > -1; i--) {
            if (get_value(data[i], idx_attr) < split_point) {
              left.push_back(data[i]);
            } else {
              right.push_back(data[i]);
            }
          }

          left_node = new node;
          left_node->build_tree(left, n_attr, random, current_height + 1, max_height, min_pts, ID);

          right_node = new node;
          right_node->build_tree(right, n_attr, random, current_height + 1, max_height, min_pts, ID);
        }

        uint32_t find_leafID(DataPoint<FeatType, LabelType> const & data) {
          if (left_node == nullptr) {
            return leafID;
          }

          float val = get_value(data, idx_attr);

          if (val < split_point) {
            return left_node->find_leafID(data);
          } else {
            return right_node->find_leafID(data);
          }
        }

      private:
        uint32_t idx_attr;
        double split_point;

        int32_t leafID{-1};

        node * left_node{nullptr};
        node * right_node{nullptr};

        float get_value(DataPoint<FeatType, LabelType> const & data, uint64_t idx_attr) {
          int64_t idx = locate_index(data, idx_attr);

          if ((idx >= 0) && (data.indexes[idx] == idx_attr)) {
            return data.features[idx];
           }

          return 0.0f;
        }

        int64_t locate_index(DataPoint<FeatType, LabelType> const & data, uint64_t idx_attr) {
          int64_t min = 0, max = data.indexes.size() - 1;

          if (max == -1) {
            return -1;
          }

        // Binary search
  //          while ((data.indexes[min] <= idx_attr) && (data.indexes[max] >= idx_attr)) {
          while (min <= max) {
            int64_t current = (max + min) / 2;

            if (data.indexes[current] > idx_attr) {
              max = current - 1;
            } else if (data.indexes[current] < idx_attr) {
              min = current + 1;
            } else {
              return current;
            }
          }

          if (data.indexes[max] < idx_attr) {
            return max;
          } else {
            return min - 1;
          }

        // linear search
  //          for (uint32_t idx = 0; idx < data.indexes.size(); idx++) {
  //            if (data.indexes[idx] == idx_attr) {
  //              return idx;
  //            }
  //          }

  //          return -1;
        }
    };

  template <typename FeatType, typename LabelType>
    class ik_iForest : public ik_mass<FeatType, LabelType> {
      public:
        ik_iForest(int sets, int psi) : ik_mass<FeatType, LabelType>(sets, psi) {
        }

        virtual ~ik_iForest() {
          for (node<FeatType, LabelType> * tree : trees) {
            delete tree;
          }

          trees.clear();
        }

        virtual void transform(DataPoint<FeatType, LabelType> const & data, Eigen::VectorXf & zt) override {
//          for (int i = 0; i < this->sets_; i++) {
//            uint32_t id = trees[i]->find_leafID(data);
//            zt[i * this->psi_ + id] = 1.0;
//          }

          for (int i = 0; i < this->sets_; i++) {
            uint32_t id = trees[i]->find_leafID(data);
            zt[i] = id;
          }
        }

      protected:
        virtual void build_model(std::vector<DataPoint<FeatType, LabelType>> & data_points, std::mt19937 & mt_random) override {
          std::cout << "Building iForest ... " << std::endl;

          int max_height = int(std::ceil(std::log2(this->psi_)));

          double start = get_current_time();

          for (int i = 0; i < this->sets_; i++) {
            std::cout << "*" << std::flush;

            std::vector<DataPoint<FeatType, LabelType>> psi_data_points;

            for (int j = 0; j < this->psi_; j++) {
              psi_data_points.push_back(data_points.back());
              data_points.pop_back();
            }

            int leafID = 0;

            node<FeatType, LabelType> * tree = new node<FeatType, LabelType>;
            tree->build_tree(psi_data_points, this->attr_, mt_random, 0, max_height, 1, leafID);
            trees.push_back(tree);
          }

          double stop = get_current_time();

          std::cout << std::endl;
          std::cout << "Build iForest model time: " << (stop - start) << std::endl;
        }

      private:
        std::vector<node<FeatType, LabelType> *> trees;
    };

}

#endif // IK_IFOREST_H
