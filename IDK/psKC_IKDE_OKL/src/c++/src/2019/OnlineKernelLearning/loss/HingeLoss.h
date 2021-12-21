/*************************************************************************
	> File Name: HingeLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/8/18 星期日 16:58:22
	> Functions: Hinge Loss function, for SVM
 ************************************************************************/

#ifndef HEADER_HINGE_LOSS
#define HEADER_HINGE_LOSS

#include "LossFunction.h"

namespace SOL {
	template <typename FeatType, typename LabelType>
	class HingeLoss: public LossFunction<FeatType, LabelType> {
		public:
			virtual  float GetLoss(LabelType label, float predict) {
                return (std::max)(0.0f, 1.f - predict * label);
			}

            virtual  float GetGradient(LabelType label, float predict) {
                if (this->GetLoss(label,predict) > 0)
                    return (float)(-label);
                else
					return 0;
			}
	};
}

#endif
