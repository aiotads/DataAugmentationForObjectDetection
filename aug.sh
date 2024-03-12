#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 sourcepath targetpath"
    exit 1
fi

SOURCE_PATH=$1
TARGET_PATH=$2

cp ${SOURCE_PATH}/* ${TARGET_PATH}/

python3 augCli.py -p ${SOURCE_PATH} -o ${TARGET_PATH} -f V
python3 augCli.py -p ${SOURCE_PATH} -o ${TARGET_PATH} -f H
python3 augCli.py -p ${SOURCE_PATH} -o ${TARGET_PATH} -r 20
python3 augCli.py -p ${SOURCE_PATH} -o ${TARGET_PATH} -hsv 30 30 30
python3 augCli.py -p ${SOURCE_PATH} -o ${TARGET_PATH} -b 5
python3 augCli.py -p ${SOURCE_PATH} -o ${TARGET_PATH} -br 50
python3 augCli.py -p ${SOURCE_PATH} -o ${TARGET_PATH} -d 100
python3 augCli.py -p ${SOURCE_PATH} -o ${TARGET_PATH} -scale
python3 augCli.py -p ${SOURCE_PATH} -o ${TARGET_PATH} -mv_dn

# mosaic
python3 augCli.py -p ${TARGET_PATH} -o ${TARGET_PATH} -m