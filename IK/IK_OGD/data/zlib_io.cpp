/*************************************************************************
  > File Name: zlib_io.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Wed 06 Nov 2013 04:15:51 PM
  > Descriptions: read and write file in default zlib format
 ************************************************************************/

#include "zlib_io.h"
#include "../common/init_param.h"

#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#  include <fcntl.h>
#  include <io.h>
#  define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#  define SET_BINARY_MODE(file)
#endif


#include <iostream>

using namespace std;

namespace SOL{
    bool zlib_io::open_file(const char* filename, const char* mode){
        if (this->alloc_buf() == false){
            this->free_buf();
            return false;
        }

        this->close_file();
        switch(mode[0]){
            case 'w':
                /* allocate deflate state */
                strm.zalloc = Z_NULL;
                strm.zfree = Z_NULL;
                strm.opaque = Z_NULL;
                if(deflateInit(&strm,zlib_deflate_level) != Z_OK)
                    return false;
                rw_mode = mode_write; 
                this->de_avail_count = zlib_buf_size;
                this->cur_de_pos = this->de_data; 
                break;
            case 'r':
                /* allocate deflate state */
                strm.zalloc = Z_NULL;
                strm.zfree = Z_NULL;
                strm.opaque = Z_NULL;
                if(inflateInit(&strm) != Z_OK)
                    return false;
                strm.avail_in = 0;
                strm.next_in = NULL;

                rw_mode = mode_read;
                this->de_avail_count = 0;
                this->cur_de_pos = this->de_data + zlib_buf_size;

                break;
            default:
                cerr<<"unrecognized file open mode!"<<endl;
                return false;
        }
        
        file = fopen(filename, mode);
        if (file == NULL){
            cerr<<"open file failed!"<<endl;
            return false;
        }
        if (this->good() != 0){
            this->close_file();
            return false;
        }
        return true;
    }

    // bind_stdin: bind the input to stdin
    bool zlib_io::open_stdin(){
        file = stdin;
        return true;
    }

    // bind_stdin: bind the output to stdout
    bool zlib_io::open_stdout(){
        file = stdout;
        return true;
    }
    void zlib_io::close_file(){
        if (file != NULL){
            if (rw_mode == mode_write){
                if (this->finalize_write() != 0){
                    /* clean up and return */
                    (void)deflateEnd(&strm);
                }
            }
            else if (rw_mode == mode_read){
                /* clean up and return */
                (void)inflateEnd(&strm);
            }

            fclose(file);
        }
        file = NULL;
    }

    bool zlib_io::alloc_buf(){
        if (this->en_data == NULL){
            try{
                this->en_data = new unsigned char[zlib_buf_size];
            }catch(std::bad_alloc &ex){
                cerr<<"allocate memory for encoded buffer failed\n";
                cerr<<ex.what()<<endl;
                return false;
            }
        }
        if (this->de_data == NULL){
            try{
                this->de_data = new unsigned char[zlib_buf_size];
            }catch(std::bad_alloc &ex){
                cerr<<"allocate memory for decoded buffer failed\n";
                cerr<<ex.what()<<endl;
                return false;
            }
        }
        return true;
    }

    void zlib_io::free_buf(){
        if (this->en_data != NULL){
            delete []this->en_data;
            this->en_data = NULL;
        }
        if (this->de_data != NULL){
            delete []this->de_data;
            this->de_data = NULL;
        }
    }

    void zlib_io::rewind(){
        if (file != NULL){
            std::rewind(file);
            this->cur_de_pos = this->de_data;
            this->de_avail_count = 0;
        }
    }

    /**
     * good : test if the io is good
     *
     * @Return: zero if correct, else zero code
     */
    int zlib_io::good(){
        if (file == NULL)
            return -1;
        return ferror(file);
    }

    /**
     * read_data : read the data from file
     *
     * @Param dst: container to place the read data
     * @Param length: length of data of read in bytes
     *
     * @Return: true if succeed
     */
    bool zlib_io::read_data(char* dst, size_t len){
        while (this->de_avail_count < len){
            memcpy(dst, this->cur_de_pos, this->de_avail_count);
            len -= this->de_avail_count;
            dst += this->de_avail_count;
            //this->buf_in_pos += this->buf_in_have; //can be ignored
            this->de_avail_count = 0; //can be ignored

            if (strm.avail_in == 0){
                strm.avail_in = fread(this->en_data, 1,zlib_buf_size,this->file);
                if (ferror(this->file)) {
                    (void)inflateEnd(&strm);
                    cerr<<"unexpected error occured when loading cache!"<<endl;
                    return false;
                }
                if (strm.avail_in == 0){
                    return false;
                }
                strm.next_in = this->en_data;
            }

            /* run inflate() */
            strm.avail_out = zlib_buf_size;
            strm.next_out = this->de_data; 
            int ret = inflate(&strm, Z_NO_FLUSH);
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            switch (ret) {
                case Z_NEED_DICT:
                    ret = Z_DATA_ERROR;     /* and fall through */
                case Z_DATA_ERROR:
                case Z_MEM_ERROR:
                    (void)inflateEnd(&strm);
                    cerr<<"error occured when parsing file!"<<endl;
                    return false;
            }
            this->de_avail_count = zlib_buf_size - strm.avail_out;

            this->cur_de_pos = this->de_data;
            if (this->cur_de_pos == 0){
                cerr<<"load compressed content failed!"<<endl;
                return false;
            }
        }
        memcpy(dst,this->cur_de_pos, len);
        this->cur_de_pos += len;
        this->de_avail_count -= len;
        //len -= len; //can be ignored
        //dst += len; //can be ignored
        return true;
    }

    /**
     * read_line : read a line from disk
     *
     * @Param dst: container to place the read data
     * @Param dst_len: length of dst
     *
     * @Return: size of data read in bytes
     */
    char* zlib_io::read_line(char* &dst, size_t &dst_len){
        printf("error: no read line is supported in zlib io\n");
        return NULL;
    }

    /**
     * write_data : write content to disk
     *
     * @Param src: source of the data
     * @Param length: length to write the data
     *
     * @Return: true if succeed
     */
    bool zlib_io::write_data(char* src, size_t len){
        while(this->de_avail_count < len){
            memcpy(this->cur_de_pos,src, this->de_avail_count);
            src += this->de_avail_count;
            len -= this->de_avail_count;

            this->strm.avail_in = zlib_buf_size;
            this->strm.next_in = this->de_data;

            // run deflate()
            do {
                strm.avail_out = zlib_buf_size;
                strm.next_out = this->en_data;
                int ret = deflate(&(this->strm), Z_NO_FLUSH);   //no bad return value 
                assert(ret != Z_STREAM_ERROR);  // state not clobbered 
                unsigned int have = zlib_buf_size - this->strm.avail_out;
                if (fwrite(this->en_data, 1, have,this->file) != have 
                        || ferror(this->file)) {
                    (void)deflateEnd(&(this->strm));
                    cerr<<"unexpected error occured when writing file!"<<endl;
                    return false;
                }
            } while (this->strm.avail_out == 0);
            assert(this->strm.avail_in == 0);     // all input will be used 

            this->de_avail_count = zlib_buf_size;
            this->cur_de_pos = this->de_data;
        }

        memcpy(this->cur_de_pos,src, len);
        this->cur_de_pos += len;
        this->de_avail_count -= len;
        this->strm.avail_in += len;

        return true;
    }
    /**
     * finalize_write : finalize write of deflate
     *
     * @Return: 0 if ok
     */
    int zlib_io::finalize_write(){
        this->strm.next_in = this->de_data;

        // run deflate()
        do {
            strm.avail_out = zlib_buf_size;
            strm.next_out = this->en_data;
            int ret = deflate(&(this->strm),Z_FINISH);   //no bad return value 
            assert(ret != Z_STREAM_ERROR);  // state not clobbered 
            unsigned int have = zlib_buf_size - this->strm.avail_out;
            if (fwrite(this->en_data, 1, have,this->file) != have 
                    || ferror(this->file)) {
                (void)deflateEnd(&(this->strm));
                cerr<<"unexpected error occured when writing file!"<<endl;
                return -1;
            }
        } while (this->strm.avail_out == 0);
        assert(this->strm.avail_in == 0);     // all input will be used 
        (void)deflateEnd(&(this->strm));
        return 0;
    }

}



