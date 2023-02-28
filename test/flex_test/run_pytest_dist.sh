#!/bin/bash
TEST_FOLDER=.
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

SEARCH_FOLDER=`pwd`
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
        if [ ! -f "test_case.py" -a -f 'host.py' -a -f 'guest.py' -a -f 'coordinator.py' ]; then
            cat <<END > test_case.py
import os
from . import host
from . import guest
from . import coordinator

def test_case():
    ROLE = os.getenv('ROLE')
    if ROLE == 'HOST':
        return host.test()
    if ROLE == 'GUEST':
        return guest.test()
    if ROLE == 'COORDINATOR':
        return coordinator.test()
END
        elif [ ! -f "test_case.py" -a -f 'host.py' -a -f 'guest.py' ]; then
            cat <<END > test_case.py
import os
from . import host
from . import guest

def test_case():
    ROLE = os.getenv('ROLE')
    if ROLE == 'HOST':
        return host.test()
    if ROLE == 'GUEST':
        return guest.test()
END
        fi
    done
}

echo "SEARCH_FOLDER:" $SEARCH_FOLDER
(cd $SEARCH_FOLDER; recursivedir)

pytest -rA $SEARCH_FOLDER --ignore=ionic_bond
find $SEARCH_FOLDER -name "test_case.py" -exec rm {} \;
find $SEARCH_FOLDER -name "__init__.py" -exec rm {} \;
touch $TEST_FOLDER/__init__.py
