#ifndef IK_MASS_H
#define IK_MASS_H

#include "ik_base.h"

#include <random>
#include <set>

namespace SOL {

  template <typename FeatType, typename LabelType>
    class ik_mass : public ik_base<FeatType, LabelType> {
      public:
        ik_mass(int sets, int psi) : sets_(sets), psi_(psi) {
        }

        virtual void build_model(Params const & param, DataSet<FeatType, LabelType> & dataset) override {
          get_inst_attr_count(dataset);

          if (param.ik_ol_init_block != -1) {
            std::cout << "Reseting the number of training instances to " << param.ik_ol_init_block << "." << std::endl;
            inst_ = param.ik_ol_init_block;
          }

          std::mt19937 mt_random{42};
          std::uniform_int_distribution<> uid = std::uniform_int_distribution<>(0, inst_ - 1);

          std::vector<uint64_t> idx;
          get_sub_sampled_set(idx, uid, mt_random);

          std::cout << "psi   : " << psi_ << std::endl;
          std::cout << "sets  : " << sets_ << std::endl;
          std::cout << "# idx : " << idx.size() << std::endl;

          std::vector<DataPoint<FeatType, LabelType>> data_points;
          get_data_points(dataset, data_points, idx);

          std::cout << "# data: " << data_points.size() << std::endl;

          build_model(data_points, mt_random);
        }

      protected:
        int sets_{100};
        int psi_{8};

        uint32_t attr_{0};
        uint32_t inst_{0};

        virtual void build_model(std::vector<DataPoint<FeatType, LabelType>> & data_points, std::mt19937 & mt_random) = 0;

      private:
        void get_data_points(DataSet<FeatType, LabelType> & dataSet, std::vector<DataPoint<FeatType, LabelType>> & data_points, std::vector<uint64_t> & idx) {
          std::vector<uint64_t> sorted_idx(idx.size());
          std::size_t n{0};
          std::generate(std::begin(sorted_idx), std::end(sorted_idx), [&]{ return n++; });
          std::sort(std::begin(sorted_idx), std::end(sorted_idx), [&](std::size_t i1, std::size_t i2) { return idx[i1] < idx[i2]; });

          if (!dataSet.Rewind()) {
            std::cerr << "Failed to rewind file. Terminating ..." << std::endl;
            exit(1);
          }

          data_points.resize(idx.size());

          std::size_t count = 0;
          std::size_t next_idx = 0;

          while (true) {
            DataChunk<FeatType,LabelType> const & chunk = dataSet.GetChunk();

            if (chunk.dataNum  == 0) {
              break;
            }

            for (size_t i = 0; i < chunk.dataNum; i++) {
              DataPoint<FeatType, LabelType> const & data = chunk.data[i];

              while ((next_idx < sorted_idx.size()) && (count == idx.at(sorted_idx.at(next_idx)))) {
                data_points[sorted_idx.at(next_idx)] = data.clone();
                next_idx++;
              }

              count++;
            }

            dataSet.FinishRead();
          }

          std::cout << "count: " << count << ", next_idx: " << next_idx << std::endl;
        }

        void get_inst_attr_count(DataSet<FeatType, LabelType> & dataSet) {
          std::cout << "Data count (start): " << dataSet.size() << std::endl;

          if (!dataSet.Rewind()) {
            std::cerr << "Failed to rewind file. Terminating ..." << std::endl;
            exit(1);
          }

          attr_ = 0;

          while (true) {
            DataChunk<FeatType,LabelType> const & chunk = dataSet.GetChunk();

            if (chunk.dataNum  == 0) {
              break;
            }

            for (size_t i = 0; i < chunk.dataNum; i++) {
              DataPoint<FeatType, LabelType> const & data = chunk.data[i];
              attr_ = std::max(attr_, data.dim());
            }

            dataSet.FinishRead();
          }

          inst_ = dataSet.size();
          std::cout << "# attr: " << attr_ << std::endl;

          std::cout << "Data count (end): " << dataSet.size() << std::endl;
        }

        void get_sub_sampled_set(std::vector<uint64_t> & idx, std::uniform_int_distribution<> & uid, std::mt19937 & mt_random) {
          std::set<uint64_t> sampling_without_replacement;

          for (int i = 0; i < sets_; i++) {
            std::vector<uint64_t> new_idx = sub_sampling(uid, mt_random, sampling_without_replacement);

            for (uint64_t val : new_idx) {
              idx.push_back(val);
            }
          }
        }

        std::vector<uint64_t> sub_sampling(std::uniform_int_distribution<> & uid, std::mt19937 & mt_random, std::set<uint64_t> & sampling_without_replacement) {
          std::vector<uint64_t> list;

          int min_psi = std::min(psi_, int(inst_));

          for (int i = 0; i < min_psi; i++) {
            if (sampling_without_replacement.size() >= inst_) {
              sampling_without_replacement.clear();
              sampling_without_replacement.insert(list.begin(), list.end());
            }

            uint64_t random_index = uid(mt_random);

            while (sampling_without_replacement.find(random_index) != sampling_without_replacement.end()) {
              random_index = uid(mt_random);
            }

            list.push_back(random_index);
            sampling_without_replacement.insert(random_index);
          }

          return list;
        }

    };

}

#endif // IK_MASS_H
