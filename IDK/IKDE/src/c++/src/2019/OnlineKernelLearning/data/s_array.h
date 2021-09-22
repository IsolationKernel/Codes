/*************************************************************************
  > File Name: s_array.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/9/19 15:14:53
  > Functions: customized array
 ************************************************************************/

#pragma once 

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <cstring>


namespace SOL {
    //the difference of s_array with vector is that vector copies the data, while
    //s_array only copies the pointer and increase counter
    template <typename T> class s_array {
        public:
            T* begin; //point to the first element
            T* end; //point to the next postion of the last element
            size_t capacity; //capacity of the array
            int *count;

            T first() const {return *begin;}
            T last() const {return *(end - 1);}
            T pop() {return *(--end);}
            bool empty() const {return begin == end;}
            size_t size() const {return end - begin;}
            T& operator[] (size_t i) {return begin[i];}
            const T& operator[] (size_t i) const { return begin[i];}

			void allocate(size_t new_size){
				T* new_begin = NULL;
				try{
					new_begin = new T[new_size];
				}catch(std::bad_alloc &ex){
					std::cerr<<ex.what();
					std::cerr<<" realloc of "<< new_size
						<<" failed in resize(). out of memory? in file " 
						<<__FILE__<<" line "<<__LINE__<<std::endl;
					exit(1);
				}
				if (new_begin == NULL && sizeof(T) *  new_size > 0) {
					std::cerr<<"realloc of "<< new_size
						<<" failed in resize(). out of memory?\n" 
						<<__FILE__<<"\n"<<__LINE__<<std::endl;
					exit(1);
				}

				size_t old_len = this->size();
				//copy data
				memcpy(new_begin,begin,sizeof(T) * old_len);
				if (begin != NULL)
					delete []begin;
				begin = new_begin;
				end = begin + old_len;
				capacity = new_size;
			}

			void resize(size_t newSize) {
				if (capacity < newSize){ //allocate more memory
					this->allocate(newSize);
				}
				end = begin + newSize;
			}
			void erase(void) { resize(0); }

			void push_back(const T& elem) {
				size_t old_len = size();
				if (old_len == capacity) {//full array
					this->allocate(2 * old_len + 3);
				}
				*(end++) = elem;
			}

			void reserve(size_t new_size){
				if(this->capacity < new_size){
					size_t alloc_size = this->capacity;
					do{
						alloc_size = 2 * alloc_size + 3;
					}while(alloc_size < new_size);
					this->allocate(alloc_size);
				}
			}

			s_array<T>& operator= (const s_array<T> &arr) {
				if (this->count == arr.count)
					return *this;
				this->release();

				this->begin =arr.begin;
				this->end = arr.end;
				this->capacity = arr.capacity;
				this->count = arr.count;
				++(*count);
				return *this;
			}

			//reset all the elements in the array to zero
			void zeros(){
				memset(this->begin, 0, sizeof(T) * this->size());
			}
			//reset all the elements in the array to zero
			void zeros(T* iter_begin, T* iter_end){
				memset(iter_begin, 0, sizeof(T) * (iter_end - iter_begin));
			}

			//set the elements in the array to val
			void set_value(const T& val){
				T* p = this->begin;
				while(p < this->end){
					*p = val;
					p++;
				}
			}
			//set the elements in the given range to the val
			void set_value(T* iter_begin, T* iter_end, const T& val){
				while(iter_begin < iter_end){
					*iter_begin = val;
					iter_begin++;
				}
			}

			void release() {
				--(*count);
				if (*count == 0) {
					if (this->begin != NULL)
						delete []this->begin;
					delete this->count;
				}
				this->begin = NULL;
				this->end = NULL;
				this->capacity = 0;
				this->count = NULL;
			}

			s_array() {
				begin = NULL; end = NULL; count = NULL; capacity = 0;
				count = new int;
				*count = 1;
			}
			s_array(const s_array &arr) {
				this->begin =arr.begin;
				this->end = arr.end;
				this->capacity = arr.capacity;
				this->count = arr.count;
				++(*count);
			}

			~s_array() { this->release(); }
	};
}
