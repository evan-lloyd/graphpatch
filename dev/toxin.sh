#!/bin/bash
# Run a command inside a Tox environment
# from https://gist.github.com/dpryden/92a9a94ed21207bba549bbe7ac41ca9f

if ! [[ -d .tox ]]; then
    echo 'Cannot find .tox in this directory!'
    exit 1
fi
if [[ "$1" == '-e' ]]; then
    toxenv="$2"
    shift 2
else
    toxenv="$(cd .tox && echo py* | awk '{print $NF}')"
fi
toxdir="$(cd .tox && pwd)"
bindir="$toxdir/$toxenv/bin"
activate_script="$bindir/activate"
if ! [[ -f $activate_script ]]; then
    printf 'Cannot find tox env "%s" in current directory!\n' "$toxenv"
    exit 1
fi
if [[ "$1" == "" ]]; then
    PATH="$bindir:$PATH" /bin/bash --rcfile "$activate_script"
else
    PATH="$bindir:$PATH" /bin/bash --rcfile "$activate_script" -c "$*"
fi
