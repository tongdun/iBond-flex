#!/bin/bash
TEST_FOLDER=..
export PYTHONPATH=$PYTHONPATH:$TEST_FOLDER/..

SEARCH_FOLDER=`pwd`/..
if [ $# -eq 1 ]; then
    SEARCH_FOLDER=$1
fi

find $SEARCH_FOLDER -name "__init__.py" -exec rm {} \;
touch $TEST_FOLDER/__init__.py
