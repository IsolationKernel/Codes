/*************************************************************************
  > File Name: comp.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 07 Nov 2013 11:01:37 PM
  > Descriptions: compression algorithms
 ************************************************************************/

#ifndef HEADER_COMP_ALGO
#define HEADER_COMP_ALGO

#include "DataPoint.h"
#include <assert.h>
#include <stdint.h>

namespace SOL{

    inline uint32_t ZigZagEncode(int32_t n) { 
        uint32_t ret = (n << 1) ^ (n >> 31);
        return ret;
    }
    inline int32_t ZigZagDecode(uint32_t n) { 
        return (n >> 1) ^ -static_cast<int32_t>(n & 1); 
    }

    //encode  an unsigned int with run length encoding
    //if encode signed int, first map it to unsigned with ZigZag Encoding
    inline void run_len_encode(s_array<char> &codes, uint32_t i){ 
        // store an int 7 bits at a time.
        while (i >= 128)    {
            codes.push_back((i & 127) | 128);
            i = i >> 7;
        }
        codes.push_back((i & 127));
    }

    inline char* run_len_decode(char* p, uint32_t& i) { // read an int 7 bits at a time.
        size_t count = 0;
        while(*p & 128)\
            i = i | ((*(p++) & 127) << 7*count++);
        i = i | (*(p++) << 7*count);
        return p;
    }

    
    /**
     * comp : compress the index list, note that the indexes must be sorted from small to big
     *  Note: the function will not erase codes by iteself
     *
     * @Param indexes: indexes to be encoded
     * @Param codes: ouput codes
     */
    inline void comp_index(const s_array<uint32_t>& indexes, s_array<char> &codes){
        uint32_t last = 0;
        size_t featNum = indexes.size();
        for (size_t i = 0; i< featNum; i++) {
            run_len_encode(codes,indexes[i] - last);
            last = indexes[i];
        }
    }

    /**
     * decomp_index : de-compress the codes to indexes
     *
     * @Param codes: input codes 
     * @Param indexes: output indexes
     */
    inline void decomp_index(s_array<char> &codes, s_array<uint32_t> &indexes){
        indexes.erase();
        uint32_t last = 0;
        uint32_t index = 0;

        char* p = codes.begin;
        while(p < codes.end){
            index = 0;
            p = run_len_decode(p,index);
            index += last;
            last = index;
            indexes.push_back(index);
        }
        assert(p == codes.end );
    }
}
#endif
