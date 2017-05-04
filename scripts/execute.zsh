#!/usr/bin/env zsh

ORIG_DIR=$PWD

# Set default options
typeset -A opts
opts[-p]=0
opts[-d]=0
opts[--inputs]=$ORIG_DIR/inputs

zparseopts -K -A opts p: d: -inputs: -program: -config:

PROGRAM=$PROGRAM
PLATFORM=$opts[-p]
DEVICE=$opts[-d]
SIZE_STRING="1024"
INPUTS=$opts[--inputs]
CONFIG=$opts[--config]

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
