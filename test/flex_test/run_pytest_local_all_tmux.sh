#!/bin/bash

if [ $# -eq 1 ]; then
    SEARCH_FOLDER=$1
fi

tmux new-session -d -s pytest
tmux set-option -g base-index 1
tmux set-window-option -g pane-base-index 1
tmux split-window -v
tmux split-window -v
tmux send -t pytest:1.1 "./run_pytest_local.sh HOST $SEARCH_FOLDER" C-m
tmux send -t pytest:1.2 "./run_pytest_local.sh GUEST $SEARCH_FOLDER" C-m
tmux send -t pytest:1.3 "./run_pytest_local.sh COORDINATOR $SEARCH_FOLDER" C-m
tmux select-layout -t pytest even-vertical
tmux set-window-option -t pytest synchronize-panes
tmux attach -t pytest
