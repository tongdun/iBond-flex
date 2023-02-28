#!/bin/bash
TEST_FOLDER=..
export COMMU_LOCALTEST=TRUE
export PYTHONPATH=$PYTHONPATH:$TEST_FOLDER/..
export ROLE=$1

SEARCH_FOLDER=`pwd`/..
if [ $# -eq 2 ]; then
    SEARCH_FOLDER=$2
fi

recursivedir() {
    for d in *; do
        if [ -d "$d" ]; then
            (cd -- "$d" && recursivedir)
        fi
        if [ ! -f "__init__.py" ]; then
            touch __init__.py
        fi
    done
}

echo "SEARCH_FOLDER:" $SEARCH_FOLDER
(cd $SEARCH_FOLDER; recursivedir)

if [[ $ROLE =~ HOST ]]; then
    echo HOST RUNNING
    pytest -rA $SEARCH_FOLDER -k "not test_1_out_n_server and not test_party_B and not test_party_C and not test_guest and not test_coordinator"
elif [[ $ROLE =~ GUEST ]]; then
    echo GUEST RUNNING
    pytest -rA $SEARCH_FOLDER -k "not test_1_out_n_client and not test_party_A and not test_party_C and not test_host and not test_coordinator"
elif [[ $ROLE =~ COORDINATOR ]]; then
    echo COORDINATOR RUNNING
    pytest -rA $SEARCH_FOLDER -k "not test_1_out_n_client and not test_1_out_n_server and not test_party_A and not test_party_B and not test_host and not test_guest"
fi
