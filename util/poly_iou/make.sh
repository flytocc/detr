#!/usr/bin/env bash

echo "Building ops ..."
rm -rf build
rm -f _C.*
python setup.py build_ext --inplace
rm -rf build
