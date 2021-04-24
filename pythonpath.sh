function join() {
    local IFS=$1
    shift
    echo "$*"
}
mystring=$(join ':' $PWD/*)

export PYTHONPATH="$mystring:$PYTHONPATH"