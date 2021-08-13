function join() {
    local IFS=$1
    shift
    echo "$*"
}

paths_to_append=$(join ':' $PWD/src/{.,features,models,omni,visualization})

export PYTHONPATH="$PYTHONPATH:$paths_to_append"
echo $PYTHONPATH
