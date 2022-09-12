set -exu

dataset=$1
num_shuffles=$2
shuffle_data_path=$3

mkdir -p $shuffle_data_path

data_name=`basename $dataset`
seed=40
rseed=$seed
pshuf() { perl -MList::Util=shuffle -e "srand($1); print shuffle(<>);" "$2"; }
for i in `seq 1  $num_shuffles`
do
    shuffled_data="${shuffle_data_path}/${data_name%%.*}_${i}.${data_name#*.}"
    if ! [ -f $shuffled_data ]; then
        echo "Shuffling $dataset > $shuffled_data"
        pshuf $rseed $dataset > $shuffled_data
    fi
    rseed=$((seed + i))
done