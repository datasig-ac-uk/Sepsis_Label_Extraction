function join() {
    local IFS=$1
    shift
    echo "$*"
}
mystring=$(join ':' $PWD/src/*)

export PYTHONPATH="$mystring:$PYTHONPATH"
echo $PYTHONPATH