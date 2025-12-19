if [ $# -ne 1 ]; then
  echo "scripts/prepare_dataset.sh: <EEG_level>"
  exit 1
fi

EEG_level=$1


echo "This scirpt construct .pickle files from .mat files from ZuCo dataset."
echo "This script also generates tenary sentiment_labels.json file for ZuCo task1-SR v1.0 and ternary_dataset.json from filtered StanfordSentimentTreebank"
echo "Note: the sentences in ZuCo task1-SR do not overlap with sentences in filtered StanfordSentimentTreebank "
echo "Note: This process can take time, please be patient..."

if [ "$EEG_level" == "rawEEG" ]; then
    python3 data/prepare_zuco_dataset/dataset_mat_to_pickle_for_rawEEG_v1.py -t task1-SR
    python3 data/prepare_zuco_dataset/dataset_mat_to_pickle_for_rawEEG_v1.py -t task2-NR
    # python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task3\ -\ TSR
    python3 data/prepare_zuco_dataset/dataset_mat_to_pickle_for_rawEEG_v2.py
elif [ "$EEG_level" == "EEG" ]; then
    python3 data/prepare_zuco_dataset/construct_dataset_mat_to_pickle_v1.py -t task1-SR
    python3 data/prepare_zuco_dataset/construct_dataset_mat_to_pickle_v1.py -t task2-NR
    # python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task3\ -\ TSR
    python3 data/prepare_zuco_dataset/construct_dataset_mat_to_pickle_v2.py
else
  echo "You only choose rawEEG or EEG"
  exit 1
fi
# python3 ./util/get_sentiment_labels.py
# python3 ./util/get_SST_ternary_dataset.py
