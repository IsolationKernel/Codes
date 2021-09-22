/*************************************************************************
  > File Name: basic_io.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Wed 06 Nov 2013 03:49:24 PM
  > Descriptions: most basic io handler, work with FILE
 ************************************************************************/

#include "basic_io.h"

#include <cstring>
#include <cstdlib>

using namespace std;

namespace SOL{
    bool basic_io::open_file(const char* filename, const char* mode){
        this->close_file();
#if _WIN32
		errno_t ret = fopen_s(&file,filename, mode);
        if ( ret != 0){
            printf("error %d: can't open input file %s\n",ret,filename);
            return false;
        }

#else
        file = fopen(filename, mode);
        if (file == NULL){
            fprintf(stderr,"open file failed!");
            return false;
        }
#endif
        if (this->good() != 0){
            this->close_file();
            return false;
        }
        return true;
    }

    // bind_stdin: bind the input to stdin
    bool basic_io::open_stdin(){
        file = stdin;
        return true;
    }

    // bind_stdin: bind the output to stdout
    bool basic_io::open_stdout(){
        file = stdout;
        return true;
    }

    void basic_io::close_file(){
        if (file != NULL && file != stdin && file != stdout){
            fclose(file);
        }
        file = NULL;
    }

    void basic_io::rewind(){
        if (file != NULL)
            std::rewind(file);
    }
    /**
     * good : test if the io is good
     *
     * @Return: zero if correct, else zero code
     */
    int basic_io::good(){
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
    bool basic_io::read_data(char* dst, size_t length){
        return fread(dst, 1, length, file) == length;
    }

    /**
     * read_line : read a line from disk
     *
     * @Param dst: container to place the read data
     * @Param dst_len: length of dst
     *
     * @Return: size of data read in bytes
     */
    char* basic_io::read_line(char* &dst, size_t &dst_len){
        size_t len;
        if(fgets(dst,dst_len,file) == NULL)
            return NULL;
        while(strrchr(dst,'\n') == NULL) {
            dst_len *= 2;
            dst = (char *) realloc(dst, dst_len);
            len = strlen(dst);
            if(fgets(dst+len,dst_len-len,file) == NULL)
                break;
        }
        return dst;
    }

    /**
     * write_data : write content to disk
     *
     * @Param src: source of the data
     * @Param length: length to write the data
     *
     * @Return: true of succeed
     */
    bool basic_io::write_data(char* src, size_t length){
        return fwrite(src, 1, length, file) == length;
    }
}


