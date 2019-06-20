#!/bin/bash

# copy example files
for i in ../example/Example*.py
do
    cp $i ./auto-${i##*example/}
done

# remove jupyter notebook information from py files:
for i in auto-Example*.py
do
    sed -i '/-*- coding: utf-8 -*-/d' $i
    sed -i '/<nbformat>4/{ N; d; }' $i
    sed -i '/<markdowncell>/{ N; d; }' $i
    sed -i '/<codecell>/{ N; d; }' $i
    # also removing plotting styles to keep code snippet shortish:
    sed -i '/plotting style:/d' $i
    sed -i '/plt.style.use/{ N; d; }' $i
    sed -i '/plt.rcParams/{ N; d; }' $i
done

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Example files were successfully copied and processed."
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
