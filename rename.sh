#!/bin/bash

i=0
for fi in *.jpg; do
    IFS=_
    set -o no glob
    field=($fi)
    label=${field[-2]}
    mv -i -- "$fi" $label_$i.jpg
    i=$((i+1))
done

#if [[ $file =~ .*_(.*)_(.*)_(.*)_(.*)\.csv$ ]]; then
#  product=${BASH_REMATCH[1]}
#  id=${BASH_REMATCH[2]}
#  name=${BASH_REMATCH[3]}
#  date=${BASH_REMATCH[4]}
#fi

#IFS=_
#set -o noglob
#field=($file) # split+glob  operator
#date=${field[-1]%.*}
#name=${field[-2]}
#id=${field[-3]}
#product=${field[-4]}
