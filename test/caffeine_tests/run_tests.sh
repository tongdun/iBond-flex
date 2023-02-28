#!/bin/bash

echo '--------------------------------------------------------------'
basepath=$(
   cd $(dirname $0)
   pwd
)
export PYTHONPATH=$PYTHONPATH:$basepath/../
echo PYTHONPATH=$PYTHONPATH
echo '--------------------------------------------------------------'

pytest 
