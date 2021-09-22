/*************************************************************************
  > File Name: /home/matthew/work/SOL/src/data/zlib_io.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Wed 06 Nov 2013 05:01:04 PM
  > Descriptions: 
 ************************************************************************/
#ifndef HEADER_ZLIB_IO
#define HEADER_ZLIB_IO

#include "io_interface.h"

#include <cstdio>
#include <string.h>
#include <assert.h>
#include "zlib.h"


namespace SOL{
#define ZLIB_BUF_SIZE  16348

    class zlib_io: public io_interface{
        private:
            enum RW_MODE{
                mode_null = 0,
                mode_read = 1,
                mode_write = 2,
            };

        private:
            FILE* file;
            z_stream strm;
            
            unsigned char* en_data; //encoded data
            unsigned char* de_data; //decoded data
            unsigned char* cur_de_pos;  //current read position of decoded data
            size_t de_avail_count; //available decoded data count

            int rw_mode;
        public:
            zlib_io():
                file(NULL), en_data(NULL), de_data(NULL),
                cur_de_pos(NULL), de_avail_count(0), rw_mode(mode_null){}

            ~zlib_io(){
                this->free_buf();
            }

        private:
            bool alloc_buf();
            void free_buf();
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

        private:
            /**
             * finalize_write : finalize write of deflate
             *
             * @Return: 0 if ok
             */
            int finalize_write();
    };
}

#endif
