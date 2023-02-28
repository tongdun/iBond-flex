#!/bin/bash
TEST_FOLDER=/home/tdops/cen/ibond_flex/test
export PYTHONPATH=$PYTHONPATH:$TEST_FOLDER/..
export ROLE=$1

# SZS=("100" "1000" "10000" "100000" "1000000")
SZS=(10 100) # 10000 100000 1000000)
CNT=(3 3) # 1000 1000 100)

echo '############## Start time perf test ###############'
for idx in ${!SZS[@]}; do
    echo '*********** size:' ${SZS[$idx]} ' count:', ${CNT[$idx]} '**********'
    if [[ $ROLE =~ HOST ]]; then
        (time python test_host.py ${SZS[$idx]} ${CNT[$idx]}) > _${SZS[$idx]}.log 2>&1
        tail -3 _${SZS[$idx]}.log | head -1
    fi

    if [[ $ROLE =~ GUEST ]]; then
        (time python test_guest.py ${SZS[$idx]} ${CNT[$idx]}) > _${SZS[$idx]}.log 2>&1
        tail -3 _${SZS[$idx]}.log | head -1
    fi

    if [[ $ROLE =~ COORD ]]; then
        (time python test_coordinator.py ${SZS[$idx]} ${CNT[$idx]}) > _${SZS[$idx]}.log 2>&1
        tail -3 _${SZS[$idx]}.log | head -1
    fi
done


echo '############## Start memory usage test ###############'
for idx in ${!SZS[@]}; do
    echo '*********** size:' ${SZS[$idx]} ' count:', ${CNT[$idx]} '**********'
    rm -f *.dat
    if [[ $ROLE =~ HOST ]]; then
        (mprof run --include-children python test_host.py ${SZS[$idx]} ${CNT[$idx]}) >/dev/null 2>&1
        sort -k2 -n *.dat | tail -n1
    fi

    if [[ $ROLE =~ GUEST ]]; then
        (mprof run --include-children python test_guest.py ${SZS[$idx]} ${CNT[$idx]}) >/dev/null 2>&1
        sort -k2 -n *.dat | tail -n1
    fi

    if [[ $ROLE =~ COORD ]]; then
        (mprof run --include-children python test_coordinator.py ${SZS[$idx]} ${CNT[$idx]}) >/dev/null 2>&1
        sort -k1 -n *.dat | tail -n1
    fi
done
