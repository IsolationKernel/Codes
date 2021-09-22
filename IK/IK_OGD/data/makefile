FLAGS = -g -Wall
#FLAGS = -O2 -s

LIBS=basic_io.o zlib_io.o gzip_io.o
TARGETS=test

all: $(TARGETS)

test:test.o
	g++ test.o -o test

MNISTConvert:MNISTConvert.cpp
	g++ $^ $(FLAGS) -o $@

analysis:data_analysis.o $(LIBS)
	g++ $^ -lz -o $@

Cacher:Cacher.o $(LIBS)
	g++ $^ -lz -o $@

%.o:%.cpp
	g++ -c $^ -o $@ $(FLAGS)

.PHONY: clean
clean:
	-rm $(TARGETS) *.o
