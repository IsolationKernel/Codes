/*************************************************************************
  > File Name: basic_io.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Wed 06 Nov 2013 03:44:46 PM
  > Descriptions: most basic io handler, work with FILE
 ************************************************************************/

#ifndef HEADER_BASIC_IO
#define HEADER_BASIC_IO

#include "io_interface.h"
#include <cstdio>

namespace SOL{
    class basic_io: public io_interface {
        private:
            FILE* file;

        public:
            basic_io():file(NULL){}
            virtual ~basic_io(){
                this->close_file();
            }

        public:
            virtual bool open_file(const char* filename, const char* mode);
            // bind_stdin: bind the input to stdin
            virtual bool open_stdin();
            // bind_stdin: bind the output to stdout
            virtual bool open_stdout();

            virtual void close_file();
            virtual void rewind();

            /**
             * good : test if the io is good
             *
             * @Return: zero if correct, else zero code
             */
            virtual int good();

        public:
            /**
             * read_data : read the data from file
             *
             * @Param dst: container to place the read data
             * @Param length: length of data of read in bytes
             *
             * @Return: true if succeed
             */
            virtual bool read_data(char* dst, size_t length);

            /**
             * read_line : read a line from disk
             *
             * @Param dst: container to place the read data
             * @Param dst_len: length of dst
             *
             * @Return: pointer to the read line, null if failed
             */
            virtual char* read_line(char* &dst, size_t &dst_len);
            
            /**
             * write_data : write content to disk
             *
             * @Param src: source of the data
             * @Param length: length to write the data
             *
             * @Return: true if succeed
             */
            virtual bool write_data(char* src, size_t length);
    };
}

#endif
