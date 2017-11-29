#!/bin/bash

#Training
python main.py -t -s 1

#Testing
python main.py -d 20171122-042600 -i 06 07 08 -j 06 07 08

