/*************************************************************************
  > File Name: io_handler.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Wed 06 Nov 2013 03:23:19 PM
  > Descriptions: handler for io
 ************************************************************************/

#include <cstdio>

using namespace std;
namespace SOL{
    class io_handler{
        public:
            io_handler(){}
            ~io_handler(){}

        public:
            bool open_file(const char* filename);
            void close_file();

        public:
            int read_data(unsigned char* dst, size_t length);
            int write_data(unsigned char* src, size_t length);
    };
}

