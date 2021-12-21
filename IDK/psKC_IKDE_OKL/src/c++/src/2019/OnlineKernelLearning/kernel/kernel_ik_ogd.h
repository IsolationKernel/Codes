#ifndef KERNEL_IK_OGD_H
#define KERNEL_IK_OGD_H

#include "ik_anne.h"
#include "ik_dot_product.h"
#include "ik_iforest.h"

#include "kernel_optim.h"

#include "../Params.h"

#include <Eigen>

namespace SOL {

// https://stackoverflow.com/questions/11635/case-insensitive-string-comparison-in-c
  inline bool iequals(std::string const & a, std::string const & b) {
    return std::equal(a.begin(), a.end(),
                      b.begin(), b.end(),
                      [](char a, char b) {
                          return std::tolower(a) == std::tolower(b);
                      });
  }

template <typename FeatType, typename LabelType>
  class kernel_ik_ogd : public Kernel_optim<FeatType, LabelType> {
    public:
      kernel_ik_ogd(Params const & param, DataSet<FeatType, LabelType> & dataset, LossFunction<FeatType, LabelType> & lossFunc) : Kernel_optim<FeatType, LabelType>(param, dataset, lossFunc) {
        this->id_str = "kernel_ogd_ik (w)";

        n_sets = param.ik_sets;
        psi = param.ik_psi;

        zt = new Eigen::VectorXf(n_sets);
        weights = new Eigen::VectorXf(n_sets * psi);

        for (int i = 0; i < (psi * n_sets); i++) {
          (*weights)(i) = 0.0f;
        }

        if (iequals(param.ik_model, "aNNE")) {
          ik_model = new ik_aNNE<FeatType, LabelType>(param.ik_sets, param.ik_psi);
          std::cout << "IK Model : aNNE" << std::endl;
        } else if (iequals(param.ik_model, "iForest")) {
          ik_model = new ik_iForest<FeatType, LabelType>(param.ik_sets, param.ik_psi);
          std::cout << "IK Model : iForest" << std::endl;
        } else if (iequals(param.ik_model, "DotProduct")) {
          ik_model = new ik_dot_product<FeatType, LabelType>(param.ik_sets, param.ik_psi);
          std::cout << "IK Model : Dot Product" << std::endl;
        } else {
          std::cerr << "Unknown IK model: " << param.ik_model << ". Aborting ..." << std::endl;
          exit(1);
        }

        ik_model->build_model(param, dataset);
      }

      virtual ~kernel_ik_ogd() {
        delete ik_model;
        delete weights;
        delete zt;
      }

    protected:
      virtual void begin_test(void) override {
      // nothing to do here
      }

      virtual float Predict(DataPoint<FeatType, LabelType> const & data) override {

//        for (int i = 0; i < (psi * n_sets); i++) {
//          (*zt)[i] = 0.0f;
//        }

        ik_model->transform(data, *zt);

//        for (std::size_t i = 0; i < data.indexes.size(); i++) {
//          (*zt)[data.indexes[i] - 1] = data.features[i];
//        }

//        return (*weights).dot(*zt);

        float res = 0.0f;

        for (int i = 0; i < n_sets; i++) {
          res += (*weights)[i * psi + (*zt)[i]];
        }

        return res;
      }

      virtual float UpdateWeightVec(DataPoint<FeatType, LabelType> const & x) override {
        float y = Predict(x); // this function also update the zt variable

        float gt_i = this->lossFunc->GetGradient(x.label, y);

        if (gt_i != 0) {
          float alpha = -(this->eta0 * gt_i);

//          for (int i = 0; i < (psi * n_sets); i++) {
//            (*zt)[i] = 0.0f;
//          }

//          for (std::size_t i = 0; i < x.indexes.size(); i++) {
//            (*zt)[x.indexes[i] - 1] = x.features[i];
//          }

//          (*weights) += alpha * *zt;

          for (int i = 0; i < n_sets; i++) {
            (*weights)[i * psi + (*zt)[i]] += alpha;
          }
        }

        return y;
      };

    private:
      ik_base<FeatType, LabelType> * ik_model{nullptr};

      int n_sets{100};
      int psi{8};

      Eigen::VectorXf * weights{nullptr};
      Eigen::VectorXf * zt{nullptr};
  };

}

#endif // KERNEL_IK_OGD_H
