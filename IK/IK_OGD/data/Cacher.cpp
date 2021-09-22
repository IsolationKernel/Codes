/*************************************************************************
  > File Name: test.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Mon 04 Nov 2013 09:50:06 PM
  > Descriptions: 
 ************************************************************************/
#if defined(_MSC_VER) && defined(_DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include "libsvmread.h"
#include "libsvm_binary.h"
#include "DataSet.h"

#include <fstream>
#include <cstdio>
#include <vector>

using namespace SOL;
void Usage(){
    cout<<"Usage: Cache input_file output_file [-d]"<<endl;
}

void Cache(const string &input_file, const string &output_file);
void Cache2(const string &input_file, const string &output_file);
void De_Cache(const string &input_file, const string &output_file);

int main(int argc, char** args){
	//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag( tmpFlag );
	//_CrtSetBreakAlloc(1698);  
#endif
	if (argc < 3){
		Usage();
		return 0;
	}

	string filename = args[1];
	string outFileName;
	bool cache = true;
	if (argc == 3){
		if (strcmp(args[2], "-d") == 0)
			cache = false;
		else
			outFileName = args[2];
	}
	else if (argc == 4){
		outFileName = args[2];
		if (strcmp(args[3], "-d") == 0)
			cache = false;
		else{
			cerr<<"incorrect parameter "<<args[3]<<endl;
			return -1;
		}
	}
	if (cache == true){
		
		// memory allocations take place here
		Cache2(filename, outFileName);
	
	}
	else
		De_Cache(filename, outFileName);

	return 0;
}

void Cache2(const string &input_file, const string &output_file){
	cout<<"Caching file..."<<endl;
	LibSVMReader reader(input_file);
	if (reader.OpenReading() == false){
		return;
	}
	DataSet<float, char> dt;
	dt.Load(input_file,output_file);
	size_t dataNum = 0;
	
	if(dt.Rewind()){
		while(1){
			const DataChunk<float, char> chunk = dt.GetChunk();
			dataNum += chunk.dataNum;
			if (chunk.dataNum == 0){
				dt.FinishRead();
				break;
			}
			dt.FinishRead();
		}
	}
}

void Cache(const string &input_file, const string &output_file){
	cout<<"Caching file..."<<endl;
	LibSVMReader reader(input_file);
	if (reader.OpenReading() == false){
		return;
	}
	string output_file_working = output_file + ".working";
	libsvm_binary writer(output_file_working);
	if (writer.OpenWriting() == false){
		return;
	}
	DataPoint<float, char> data;
	size_t dataNum = 0;
	size_t featNum = 0;
	while(reader.GetNextData(data) == true){
		dataNum++;
		featNum += data.indexes.size();
		if(!(data.label == 1 || data.label == -1)){
			cout<<"data index "<<dataNum<<" label = "<<(int)(data.label)<<endl;
		}
		writer.WriteData(data);
	}
	reader.Close();
	writer.Close();
	cout<<"data number: "<<dataNum<<endl;
	cout<<"feat number: "<<featNum<<endl;

	string cmd = "mv \"";
	cmd += output_file_working;
	cmd += "\" \"";
	cmd += output_file;
	cmd += "\"";
	int ret = system(cmd.c_str());
	if (ret != 0){
		cerr<<"rename file failed!"<<endl;
	}
}

void De_Cache(const string &input_file, const string &output_file){
	cout<<"De-Caching file..."<<endl;
	libsvm_binary reader(input_file);
	if (reader.OpenReading() == false){
		cerr<<"open "<<input_file<<" failed!"<<endl;
		return;
	}
	bool is_write = output_file.length() > 0 ? true : false;
	std::ofstream writer;
	if (is_write){
		writer.open(output_file.c_str(), ios::out);
		if (writer.good() == false){
			cerr<<"open output file" <<output_file<<" failed!"<<endl;
			return;
		}
	}
	DataPoint<float, char> data;
	size_t dataNum = 0;
	size_t featNum = 0;
	while(reader.GetNextData(data) == true){
		dataNum++;
		featNum += data.indexes.size();
		if (is_write){
			writer<<data.label;
			for (IndexType i = 0; i < data.dim(); i++){
				writer<<" "<<data.indexes[i]<<":"<<data.features[i];
			}
			writer<<"\n";
		}
	}
	reader.Close();
	if (is_write)
		writer.close();
	cout<<"data number: "<<dataNum<<endl;
	cout<<"feat number: "<<featNum<<endl;
}
