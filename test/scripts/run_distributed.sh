#!/bin/bash
TEST_FOLDER=..
HOSTS=("d-014010" "d-014011" "d-014012")

export PYTHONPATH=$PYTHONPATH:$TEST_FOLDER/..
NODE=`hostname`
if [[ $NODE =~ ${HOSTS[0]} ]]; then
    export ROLE=HOST
elif [[ $NODE =~ ${HOSTS[1]} ]]; then
    export ROLE=GUEST
elif [[ $NODE =~ ${HOSTS[2]} ]]; then
    export ROLE=COORDINATOR
fi
echo ROLE: $ROLE

SEARCH_FOLDER=`pwd`/..
if [ $# -eq 1 ]; then
    SEARCH_FOLDER=$1
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
    pytest -rA $SEARCH_FOLDER --ignore=$TEST_FOLDER/ionic_bond -k "not test_1_out_n_server and not test_party_B and not test_party_C and not test_guest and not test_coordinator"
elif [[ $ROLE =~ GUEST ]]; then
    echo GUEST RUNNING
    pytest -rA $SEARCH_FOLDER --ignore=$TEST_FOLDER/ionic_bond -k "not test_1_out_n_client and not test_party_A and not test_party_C and not test_host and not test_coordinator"
elif [[ $ROLE =~ COORDINATOR ]]; then
    echo COORDINATOR RUNNING
    pytest -rA $SEARCH_FOLDER --ignore=$TEST_FOLDER/ionic_bond -k "not test_1_out_n_client and not test_1_out_n_server and not test_party_A and not test_party_B and not test_host and not test_guest"
fi

find $SEARCH_FOLDER -name "__init__.py" -exec rm {} \;
touch $TEST_FOLDER/__init__.py
