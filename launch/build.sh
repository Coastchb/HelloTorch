#!/bin/bash

# run this script with: . build.sh
rm -rf !(build.sh) 
cmake3 -DCMAKE_PREFIX_PATH=/root/coastcao/HelloTorch/launch/libtorch ..
cmake3 --build . --config Release
