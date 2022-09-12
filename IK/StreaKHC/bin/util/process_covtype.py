import sys
import numpy as np

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    dim=54
    id = 0
    with open(input_file,'r') as fin:
        with open(output_file,'w') as fout:
            for line in fin:
                splt = line.strip().split(",")
                fout.write(str(id))
                fout.write(",")
                fout.write(str(splt[-1]))

                for item in splt[:-1]:
                    fout.write(",")
                    fout.write(str(item))
                fout.write("\n")
                id += 1
