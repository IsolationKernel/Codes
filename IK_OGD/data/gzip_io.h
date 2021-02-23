/*************************************************************************
  > File Name: /home/matthew/work/SOL/src/data/gzip_io.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Wed 06 Nov 2013 04:02:12 PM
  > Descriptions: 
 ************************************************************************/
#ifndef HEADER_GZIP_IO
#define HEADER_GZIP_IO

#include "io_interface.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "zlib.h"

namespace SOL{
    class gzip_io: public io_interface{
        private:
            gzFile file;

        public:
            gzip_io():file(NULL){}
            ~gzip_io(){
                this->close_file();
            }


        public:
            virtual bool open_file(const char* filename, const char* mode);
            virtual void close_file();
            virtual void rewind();
            // bind_stdin: bind the input to stdin
            virtual bool open_stdin();
            // bind_stdin: bind the output to stdout
            virtual bool open_stdout();

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
