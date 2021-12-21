/*************************************************************************
  > File Name: libsvmread.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/8/18 星期日 20:25:28
  > Functions: libsvm reader
 ************************************************************************/
#pragma once

#if _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "DataReader.h"
#include "basic_io.h"
#include "parser.h"

#include <stdio.h>
#include <vector>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <limits>
#include <cmath>

using namespace std;

namespace SOL {
    template <typename FeatType, typename LabelType>
        class LibSVMReader_: public DataReader<FeatType, LabelType> { 
            private:
                string fileName;
                basic_io reader;

                char *line;
                size_t max_line_len;

            public:
                LibSVMReader_(const string &fileName) {
                    this->max_line_len = 4096;
                    this->fileName = fileName;
                    line = (char *) malloc(max_line_len*sizeof(char));
                }
                ~LibSVMReader_() {
                    this->Close();
                    if (line != NULL)
                        free(line);
                }

                //////////////////online mode//////////////////
            public:
                virtual bool OpenReading() {
                    this->Close();
                    return reader.open_file(this->fileName.c_str(), "r");
                }
                virtual void Rewind() {
                    reader.rewind();
                }
                virtual void Close() {
                    reader.close_file();
                }

                virtual inline bool Good() {
                    return reader.good() == 0 ? true: false;
                }

                virtual bool GetNextData(DataPoint<FeatType, LabelType> &data) {
                    if(reader.read_line(line, max_line_len) == NULL)
                        return false;

                    LabelType labelVal;
                    char* p = line, *endptr = NULL;
                    if (*p == '\0')
                        return false;
                    labelVal = (LabelType)parseInt(p,&endptr);
                    if (endptr == p) {
						return false;
                    }

                    data.erase();
                    IndexType index;
                    FeatType feat;
                    // features
                    while(1) {
                        p = strip_line(endptr);
                        if (*p == '\0')
                            break;
                        index = (IndexType)(parseUint(p,&endptr));
                        if (endptr == p) { //parse index failed
                            fprintf(stderr,"parse index value failed!\n%s", p);
                            return false;
                        }

                        p = endptr;
                        feat = parseFloat(p,&endptr);
                        //feat =(float)(strtod(val,&endptr));
                        if (endptr == p) {
                            fprintf(stderr,"parse feature value failed!\n");
                            return false;
                        }

                        data.AddNewFeat(index,feat);
                    }
                    data.label = labelVal;
					
                    return true;
                }
        };

    //for special definition
    typedef LibSVMReader_<float, char> LibSVMReader;
}
