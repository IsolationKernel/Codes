/*************************************************************************

	> File Name: MNISTReader.h

	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>

	> Created Time: 2013/8/18 Sunday 20:25:28

	> Functions: MNIST reader

 ************************************************************************/

#ifndef HEADER_MINST_READER
#define HEADER_MINST_READER
#include "DataReader.h"

#include <fstream>
using std::ios_base;
using std::ios;
using std::ifstream;
using std::string;

namespace SOL {

template <typename T>
int MSB2LSB(T x) {
	int y = x;
	int byteNum = sizeof(x);
	char *buf = new char[byteNum];
	char *buf1 = new char[byteNum];
	memcpy(buf, &x, byteNum);
	for (int i = 0; i < byteNum; i++)
		buf1[i] = buf[byteNum - 1 - i];
	memcpy(&y,buf1, byteNum);

	delete []buf;
	delete []buf1;
	return y;
}


    template <typename FeatType, typename LabelType>
	class MNISTReader: public DataReader<FeatType, LabelType> {
        private:
			ifstream inTrainFile, inLabelFile;
			string trainFileName, labelFileName;
			int num1, num2;

			int featDim;
			int imgNum;

			std::streamoff trainFileStartPos;
			std::streamoff labelFileStartPos;

            unsigned char* rd_buf;

	public:
		MNISTReader(const string &trainFile, const string &labelFile, 
			int digit1 = -1, int digit2 = -1):
			trainFileName(trainFile),labelFileName(labelFile),
			num1(digit1),num2(digit2) {
            featDim = 0;
            imgNum = 0;
            rd_buf = NULL;
        }

		~MNISTReader() {
            this->Close();
            if (rd_buf != NULL)
                delete []this->rd_buf;
		}

		public:
			virtual bool OpenReading() {
                this->Close();
                if (this->rd_buf != NULL)
                    delete []this->rd_buf;
                this->rd_buf = NULL;

				inTrainFile.open(trainFileName.c_str(), ios::in | ios::binary);
				if(!inTrainFile) {
					printf("can't open input file %s\n",trainFileName.c_str());
					return false;
				}

				inLabelFile.open(labelFileName.c_str(), ios::in | ios::binary);
				if(!inLabelFile) {
					printf("can't open input file %s\n",labelFileName.c_str());
					return false;
					
				}

				return this->GetFeatInfo();
			}

			virtual void Rewind() {
				if(this->inTrainFile.is_open() == true)
					this->inTrainFile.seekg(trainFileStartPos,ios_base::beg);

				if(this->inLabelFile.is_open() == true)
					this->inLabelFile.seekg(labelFileStartPos,ios_base::beg);

			}
            virtual void Close() {
                this->inTrainFile.close();
				this->inLabelFile.close();
            }
            virtual bool Good() {
                if (this->inTrainFile.good() || this->inLabelFile.good() || 
                        this->inTrainFile.eof() || this->inLabelFile.eof())
                    return true;
                return false;
            }

			virtual bool GetNextData(DataPoint<FeatType, LabelType> &data) {
				if (num1 == -1 || num2 == -1)
					return ReadData(data);
				else {
					while(ReadData(data)) {
						if (data.label  == num1) {
                            data.label = 1;
							return true;
						}
						else if (data.label  == num2) {
                            data.label = -1;
							return true;
						}
					}
					return false;
				}
			}

		private:
			bool ReadData(DataPoint<FeatType, LabelType> &data) {
				if (!inTrainFile.good() || !inLabelFile.good())
					return false;

				data.erase();
				//get next label
				char label;
				inLabelFile.read((char*)&label,sizeof(char)); 
				//get feature
				inTrainFile.read((char*)this->rd_buf,sizeof(unsigned char) *featDim);

                for (int i = 0; i < featDim; i++) {
                    if (this->rd_buf[i] != 0)
                        data.AddNewFeat(i + 1,this->rd_buf[i]);
                }
                data.label = label;
				return true;
			}

		
		bool GetFeatInfo() {
			inTrainFile.seekg(0,ios_base::beg);
			inLabelFile.seekg(0,ios_base::beg);

			if(!inTrainFile || !inLabelFile)
				return false;

			//label file
			//Load header
			int magicNum(0);
			inLabelFile.seekg(0,ios_base::beg);
			inLabelFile.read((char*)&magicNum, sizeof(int));
			magicNum = MSB2LSB(magicNum);
			if (magicNum != 2049) {
				std::cerr<<"Incorrect file!"<<std::endl;
				inLabelFile.close();
				return false;
			}

			int imgNum_label(0);
			inLabelFile.read((char*)&imgNum_label, sizeof(int));
			imgNum_label = MSB2LSB(imgNum_label);

			//Train File
			//Load header
			inTrainFile.read((char*)&magicNum, sizeof(int));
			magicNum = MSB2LSB(magicNum);
			if (magicNum != 2051) {
				std::cerr<<"Incorrect file!"<<std::endl;
				inTrainFile.close();
				return false;
			}

			inTrainFile.read((char*)&imgNum, sizeof(int));
			imgNum = MSB2LSB(imgNum);
			if(imgNum != imgNum_label) {
				std::cerr<<"Label and Training set is not consistent!"<<std::endl;
				inTrainFile.close();
				return false;
			}

			int rowNum(0), colNum(0);
			inTrainFile.read((char*)&rowNum, sizeof(int));
			inTrainFile.read((char*)&colNum, sizeof(int));
			rowNum = MSB2LSB(rowNum);
			colNum = MSB2LSB(colNum);

			featDim = rowNum * colNum;

			trainFileStartPos = inTrainFile.tellg();
			labelFileStartPos = inLabelFile.tellg();

            if (featDim > 0)
                this->rd_buf = new unsigned char[featDim];

			return true;
		}
	};
}
#endif
