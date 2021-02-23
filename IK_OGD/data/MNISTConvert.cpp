/*************************************************************************
  > File Name: Convert.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 14 Nov 2013 06:49:38 PM
  > Descriptions: Convert other file formats to LIBSVM
 ************************************************************************/

#include <string>
#include <fstream>
using namespace std;

#include "MNISTReader.h"
using namespace SOL;
#define FeatType float
#define LabelType char
//define your own data reader here
#define ReaderType MNISTReader<FeatType, LabelType>

int main(int argc, char** args){ 
    if (argc != 6){
        cout<<"Usage: MNISTConvert train_file label_file digit1 digit2 output_file"<<endl;
        return 0;
    }
    string trainfile = args[1];
    string labelfile = args[2];
    int num1 = atoi(args[3]);
    int num2 = atoi(args[4]);
    string outfilename = args[5];
    //string filename = "/home/matthew/work/Data/aut/aut_train";
    ReaderType reader(trainfile, labelfile, num1, num2);
    DataPoint<FeatType, LabelType> data;
    if (reader.OpenReading() == false) {
        return -1;
    }
    ofstream outFile(outfilename.c_str(), ios::out);
    if (!outFile){
        cerr<<"open file "<<outfilename<<" failed!"<<endl;
        return -1;
    }

    size_t dataNum = 0;
    reader.Rewind();
    cout<<"converting ...... "<<endl;
    while(true) {
        if (reader.GetNextData(data) == true) {
            outFile<<(int)(data.label);
            for (size_t i = 0; i < data.indexes.size(); i++){
                outFile<<" "<<data.indexes[i]<<":"<<data.features[i];
            }
            dataNum++;
            outFile<<"\n";
        }
        else if (reader.Good() == true){
            break;
        }
        else{
            cerr<<"unexpected error occured when loading data"<<endl;
            break;
        }
    }
    reader.Close();
    outFile.close();
    cout<<"data number  : "<<dataNum<<"\n";
    return 0;
}

