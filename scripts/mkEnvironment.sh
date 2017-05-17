#!/bin/bash

# get the path of the script itself
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

# set some default values
DEFAULT_BASEPATH=$(readlink -f ${SCRIPTPATH}/../..)
# default paths
DEFAULT_LIFT_LOCATION="${DEFAULT_BASEPATH}/lift/"
DEFAULT_HARNESS_LOCATION="${DEFAULT_BASEPATH}/harness/"
DEFAULT_TUNER_LOCATION="${DEFAULT_BASEPATH}/atf/bld/examples/genericLiftKernel/"
DEFAULT_PLOTSCRIPT_LOCATION="${DEFAULT_BASEPATH}/exploration/R/violinShoc.r"

#default cl env
DEFAULT_CL_PLATTFORM=0
DEFAULT_CL_DEVICE=0

# read args
confFile=$1

printf "\nThis script will create an environment config at ${confFile}\n"

confDir=$(dirname "$confFile")
mkdir -p $confDir # create dir if nonexistent


# ask for default paths
printf "\nSet up the paths.\n"
read -e -p "Where is lift located? : " -i "$DEFAULT_LIFT_LOCATION" liftLocation
read -e -p "Where is harness located? : " -i "$DEFAULT_HARNESS_LOCATION" executorLocation
read -e -p "Where the R plot script located? : " -i "$DEFAULT_PLOTSCRIPT_LOCATION" plotscriptLocation


# ask for the openCL setup
printf "\nSet up the openCL environment\n"
read -e -i "$DEFAULT_CL_PLATTFORM" -p "Which openCL plattform do you want to use? " clPlattform
read -e -i "$DEFAULT_CL_DEVICE" -p "Which openCL device do you want to use? " clDevice

# write the file
cat <<EOF > "$confFile"
[Path]
Lift = $liftLocation
Executor = $executorLocation
Rscript = $plotscriptLocation

[OpenCL]
Platform = $clPlattform
Device = $clDevice
EOF

printf "\nConfig file created and ready to use\n\n"
