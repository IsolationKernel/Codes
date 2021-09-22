/*************************************************************************
	> File Name: SquaredHingeLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/11/27 11:30:44
	> Functions: Squared Hinge loss
 ************************************************************************/
#ifndef HEADER_SQUARE_HINGE_LOSS
#define HEADER_SQUARE_HINGE_LOSS

#include "LossFunction.h"

namespace SOL {
	template <typename FeatType, typename LabelType>
	class SquaredHingeLoss: public LossFunction<FeatType, LabelType> {
		public:
			virtual  float GetLoss(LabelType label, float predict) {
                float loss = (std::max)(0.0f, 1.f - predict * label);
                return loss * loss;
			}

            virtual  float GetGradient(LabelType label, float predict) {
                float loss = (std::max)(0.0f, 1.f - predict * label);
                if (loss > 0)
                    return -label * loss * 2.f;
                else
					return 0;
			}
	};
}

#endif
