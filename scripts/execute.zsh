#!/usr/bin/zsh

PROGRAM=dummy
PLATFORM=0
DEVICE=0
SIZE_STRING="1024"
ORIG_DIR=$PWD
INPUTS=/home/s1042579/generated_programs/inputs
CONFIG=/home/s1042579/harness/bla_interface.json

for i in $(seq 1 250)
do
  for low_level in $(find $PROGRAM -mindepth 1 -type d)
  do

    DONE_FILE="$ORIG_DIR/$low_level/done_$SIZE_STRING"
    if [[ ! -a $DONE_FILE ]]
    then
      cd $low_level

      timeout 3m harness_generic --folder $INPUTS --file $CONFIG -p $PLATFORM -d $DEVICE

      cd $ORIG_DIR
    fi
  done
done
