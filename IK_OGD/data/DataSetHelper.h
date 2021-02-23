/*************************************************************************
> File Name: DataSetHelper.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Thu 24 Oct 2013 03:33:10 PM
> Descriptions: thread function definitions
************************************************************************/
#pragma once


#include "libsvm_binary.h"
#include "thread_primitive.h"

namespace SOL{
	template <typename T1, typename T2> class DataSet;

	//load a chunk of data, return if file ended
	template <typename T1, typename T2>
	bool load_chunk(DataReader<T1, T2>* reader, DataChunk<T1,T2>&chunk){
		bool not_file_end = true;
		chunk.erase();
		while(chunk.dataNum < init_chunk_size && not_file_end == true){
			DataPoint<T1,T2> &data = chunk.data[chunk.dataNum];
			not_file_end = reader->GetNextData(data);
			if (not_file_end == true)
				chunk.dataNum++;
			else
				break;
		}
		return not_file_end;
	}

	//save chunk to disk
	template <typename T1, typename T2>
	bool save_chunk(libsvm_binary_<T1, T2> *writer, DataChunk<T1, T2>&chunk){
		size_t w_num = 0;
		while(w_num < chunk.dataNum){
			if (writer->WriteData(chunk.data[w_num]) == true)
				w_num++;
			else
				return false;
		}
		return false;
	}

	template <typename T1, typename T2>
	libsvm_binary_<T1, T2>* get_cacher(const std::string &cache_filename){
		string tmpFileName = cache_filename + ".writing";
		libsvm_binary_<T1, T2>* cacher = new libsvm_binary_<T1,T2>(tmpFileName);
		if (cacher->OpenWriting() == false){
			cerr<<"Open cache file failed!"<<endl;
			delete cacher;
			return NULL;
		}
		return cacher;
	}

	template <typename T1, typename T2>
	bool end_cache(libsvm_binary_<T1, T2>**cacher, const std::string& cache_fileName){
		string tmpFileName = (*cacher)->get_filename();
		(*cacher)->Close();
		delete *cacher;
		*cacher = NULL;	

		//rename
#if WIN32
		string cmd = "ren \"";
		cmd = cmd + tmpFileName + "\" \"";
		//in windows, the second parameter of ren should not include path
		cmd = cmd + cache_fileName.substr(cache_fileName.find_last_of("/\\") + 1) + "\"";
#else
		string cmd = "mv \"";
		cmd = cmd + tmpFileName + "\" \"";
		cmd = cmd + cache_fileName + "\"";
#endif
		
		if(system(cmd.c_str()) != 0){
			cerr<<"rename cahe file name failed!"<<endl;
			return false;
		}
		return true;
	}

	template <typename T1, typename T2>
	bool CacheLoad(DataSet<T1, T2> *dataset){
		DataReader<T1,T2>* reader = dataset->reader;
		reader->Rewind();
		if (reader->Good() == false) {
			cerr<<"reader is incorrect!"<<endl;
			return false;
		}

		libsvm_binary_<T1,T2>* writer = get_cacher<T1,T2>(dataset->cache_fileName);
		if (writer == NULL)
			return false;

		//load data
		bool not_file_end = false;
		do {
			DataChunk<T1,T2> &chunk = dataset->GetWriteChunk();
			not_file_end = load_chunk(reader, chunk);
			save_chunk(writer, chunk);
			dataset->EndWriteChunk();
		}while(not_file_end == true);

		return end_cache(&writer, dataset->cache_fileName);
	}

	template <typename T1, typename T2> 
#if WIN32
	DWORD WINAPI thread_LoadData(LPVOID param)
#else
	void* thread_LoadData(void* param)
#endif
	{
		DataSet<T1,T2>* dataset = static_cast<DataSet<T1,T2>*>(param);
		DataReader<T1,T2>* reader = dataset->reader;

		size_t pass = 0;
		if (dataset->is_cache == true){
			if(CacheLoad(dataset) == false){
				cerr<<"caching data failed!"<<endl;
				dataset->FinishParse();
				return NULL;
			}
			dataset->reader->Close();
			delete dataset->reader;
			//load cache file
			dataset->reader = new libsvm_binary_<T1,T2>(dataset->cache_fileName);
			if (dataset->reader->OpenReading() == false){
				cerr<<"load cache data failed!"<<endl;
				dataset->FinishParse();
				return NULL;
			}
			reader = dataset->reader;
			dataset->is_cache = false;
			pass++;
		}
		//load cache
		for (;pass < dataset->passNum; pass++) {
			reader->Rewind();
			if (reader->Good()) {
				bool not_file_end = false;
				do {
					DataChunk<T1,T2> &chunk = dataset->GetWriteChunk();
					not_file_end = load_chunk(reader, chunk);
					dataset->EndWriteChunk();
				}while(not_file_end == true);
			}
			else {
				cerr<<"reader is incorrect!"<<endl;
				break;
			}
		}
		dataset->FinishParse();
		return NULL;
	}
}
