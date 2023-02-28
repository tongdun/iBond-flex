#!/bin/bash

if [ $# -eq 1 ]; then
    SEARCH_FOLDER=$1
fi

(./run_pytest_local.sh HOST $SEARCH_FOLDER > _host_test.log) &
(./run_pytest_local.sh COORDINATOR $SEARCH_FOLDER > _coord_test.log) &
./run_pytest_local.sh GUEST $SEARCH_FOLDER
wait %1
wait %2
