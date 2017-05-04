#!/usr/bin/env zsh

REQUIRED_VERSION="5.1"

if [[ $ZSH_VERSION < $REQUIRED_VERSION ]]
then
  echo -n "The script might not work correctly with zsh version $ZSH_VERSION." >&2
  echo " Update to $REQUIRED_VERSION." >&2
  exit 1
fi

function check_command() {
  if ! type "$1" > /dev/null
  then
    echo "Missing command '$1'" >&2
    exit 1
  fi
}

function check_file() {
  if [[ ! -a $1 ]]
  then
    echo "Missing file '$1' provided for option $2" >&2
    exit 1
  fi
}

ORIG_DIR=$PWD

# Set default options and parse
typeset -A OPTS
OPTS[-p]=0
OPTS[-d]=0
OPTS[-i]=250
OPTS[--inputs]=$ORIG_DIR/inputs

zparseopts -K -A OPTS p: d: i: -inputs: -program: -config:

# Get parsed/default options
PROGRAM=$(realpath $OPTS[--program])
INPUTS=$(realpath $OPTS[--inputs])
CONFIG=$(realpath $OPTS[--config])
PLATFORM=$OPTS[-p]
DEVICE=$OPTS[-d]
ITERATIONS=$OPTS[-i]

# Check that program/inputs/config exist
check_command harness_generic
check_command jq

check_file $PROGRAM \'--program\'
check_file $INPUTS \'--inputs\'
check_file $CONFIG \'--inputs\'

# Get the size string used for the done file
NUM_DIFF_SIZES=$(jq '.sizes' $CONFIG | sed -e '/\[/d' -e ' /\]/d' -e 's/,//g' -e 's/ //g' | uniq | wc -l)

if [[ $NUM_DIFF_SIZES = 1 ]]
then
  SIZE_STRING=$(jq '.sizes[0]' $CONFIG)
else
  SIZE_STRING=$(jq -c '.sizes' $CONFIG | sed -e 's/\[//g' -e 's/\]//g' -e 's/,/_/g')
fi

# Try to execute
for i in $(seq 1 $ITERATIONS)
do
  for LOW_LEVEL in $(find $PROGRAM -mindepth 1 -type d)
  do

    DONE_FILE="$LOW_LEVEL/done_$SIZE_STRING"

    # If done marker is set, ignore
    if [[ ! -a $DONE_FILE ]]
    then
      cd $LOW_LEVEL

      timeout 3m harness_generic --folder $INPUTS --file $CONFIG -p $PLATFORM -d $DEVICE
    fi
  done

done
