function join() {
    local IFS=$1
    shift
    echo "$*"
}
mystring=$(join ':' $PWD/src/{.,features,models,omni,visualization})

export PYTHONPATH="$mystring:$PYTHONPATH"
echo $PYTHONPATH
