/*************************************************************************
> File Name: DataPoint.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 20:13:31
> Functions: Data Point Definition
************************************************************************/

#pragma once

#include "s_array.h"
#include "../common/init_param.h"

#include <cstring>

namespace SOL {
	/**
	*  Definitions of DataPoint: one lable, and DataPoints
	*
	* @tparam DataType
	*/
	template <typename FeatType, typename LabelType> 
	class DataPoint {
	public:
		//////////////Member Variables
		s_array<IndexType> indexes;
		s_array<FeatType> features;
		LabelType label;
		FeatType sum_sq; //sum of square

		//for copy and release control
		int *count;

		IndexType max_index; //max index, also the dimension
	public:
		DataPoint() {
			this->count = new int;
			*count = 1;
			this->max_index = 0;
			this->label = 0;
			this->sum_sq = 0;
		}

		//copy constructor
		DataPoint(const DataPoint &point) {
			this->indexes = point.indexes;
			this->features = point.features;
			this->label = point.label;
			this->count = point.count;
			this->max_index = point.max_index;
			this->sum_sq = 0;
			++(*count);
		}

		~DataPoint(){
			this->release();
		} 

		//assignment
		DataPoint<FeatType, LabelType>& operator= 
			(const DataPoint<FeatType, LabelType> &data) {
				if (data.count == this->count)
					return *this;
				this->release();

				this->indexes = data.indexes;
				this->features = data.features;
				this->label = data.label;
				this->max_index = data.max_index;
				this->sum_sq = data.sum_sq;
				this->count = data.count;
				++(*count);
				return *this;
		}
		//set new index-value pair
		void AddNewFeat(const IndexType &index, 
			const FeatType &feat) {
				this->indexes.push_back(index);
				this->features.push_back(feat);
				if(this->max_index < index){
					this->max_index = index;
				}
				this->sum_sq += feat * feat;
		}

		void erase() {
			this->indexes.erase();
			this->features.erase();
			this->max_index = 0;
			this->sum_sq = 0;
		}


		DataPoint<FeatType, LabelType> clone() const{
			DataPoint<FeatType, LabelType> newPt; 
			newPt.label = this->label;
			newPt.max_index = this->max_index;
			newPt.sum_sq = this->sum_sq;
			newPt.indexes.resize(this->indexes.size());
			memcpy(newPt.indexes.begin,this->indexes.begin, this->indexes.size() * sizeof(IndexType) );
			newPt.features.resize(this->features.size());
			memcpy(newPt.features.begin, this->features.begin, this->features.size() * sizeof(FeatType));
			return newPt;				
		}

		IndexType dim() const {return this->max_index;}

	private:
		void release() {
			--(*count); 
			if (*count == 0)
				delete count;
			this->count = NULL;
		}

	};
	template <typename FeatType, typename LabelType> 
	struct DataChunk{
		DataPoint<FeatType, LabelType> data[init_chunk_size];
		size_t dataNum;
		bool is_inuse;
		bool is_parsed;
		DataChunk *next;

		DataChunk():dataNum(0),next(NULL), is_inuse(false), is_parsed(false){
		}
		void erase() {
			for (size_t i = 0; i < dataNum; i++)
				data[i].erase();
			dataNum = 0;
		}
	};

}
