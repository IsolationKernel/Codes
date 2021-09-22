/*************************************************************************
  > File Name: data_analysis.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 24 Oct 2013 08:09:38 PM
  > Descriptions: analyse the sparsity of data
 ************************************************************************/
#include "DataPoint.h"
#include "DataReader.h"
#include "libsvmread.h"
#include "MNISTReader.h"

#include <string>
using namespace std;
using namespace SOL;

template <typename FeatType, typename LabelType>
bool Analyze(DataReader<FeatType, LabelType> *reader) {
    if (reader == NULL){
        cerr<<"data reader is emptyp!"<<endl;
        return false;
    }

	size_t max_show_count = 100000;
	size_t show_count = 1000;
    size_t dataNum = 0;
    size_t featNum = 0;
	size_t pos_num = 0;
	size_t neg_num = 0;
    IndexType max_index = 0;
    s_array<char> index_set;
    DataPoint<FeatType, LabelType> data;
    if (reader->OpenReading() == true) {
        reader->Rewind();
        while(true) {
            if (reader->GetNextData(data) == true) {
                if (data.indexes.size() == 0)
                    continue;
                if (max_index < data.dim()){
                    max_index = data.dim();
                }
                size_t prev_size = index_set.size();
                if (max_index > prev_size){
                    index_set.reserve(max_index);
                    index_set.resize(max_index);
                    //set the new value to zero
                    index_set.zeros(index_set.begin + prev_size, 
                            index_set.end);
                }
                for (size_t i = 0; i < data.indexes.size(); i++){
                    index_set[data.indexes[i] - 1] = 1;
                }

                dataNum++;
				if (data.label == 1)
					pos_num++;
				else if (data.label == -1)
					neg_num++;
				else{
					cerr<<"\nunrecognized label!"<<endl;
					break;
				}

                featNum += data.indexes.size();
                
				if (dataNum % show_count == 0){
					cerr<<"data number  : "<<dataNum<<"    ";
					cerr<<"valid dim    : "<<max_index<<"\r";
					show_count *= 2;
					show_count = show_count > max_show_count ? 
						max_show_count : show_count;
				}
			}
			else
				break;
		}
	}
	else {
		cerr<<"Can not open file to read!"<<endl;
		return false;
	}
	cerr<<"\n";
	reader->Close();
	size_t valid_dim = 0;
	for (size_t i = 0; i < index_set.size(); i++) {
		if (index_set[i] == 1)
			valid_dim++;
	}
	cout<<"data number  : "<<dataNum<<"\n";
	cout<<"feat number  : "<<featNum<<"\n";
	cout<<"valid dim    : "<<max_index<<"\n";
	cout<<"nonzero feat : "<<valid_dim<<"\n";
	cout<<"positive num	: "<<pos_num<<"\n";
	cout<<"negtive num	: "<<neg_num<<"\n";
	if (max_index > 0){
		printf("data sparsity: %.2lf%%\n",100 - valid_dim * 100.0 / max_index);
	}

	return true;
}

int main(int argc, char** args){ 
	if (argc != 2){
		cout<<"Usage: data_analysis data_file"<<endl;
		return 0;
	}
	//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag( tmpFlag );
#endif
	string filename = args[1];
	//string filename = "/home/matthew/work/Data/aut/aut_train";
	LibSVMReader reader(filename);
	if (Analyze(&reader) == false)
		cerr<<"analyze dataset failed!"<<endl;
	return 0;
}
