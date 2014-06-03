#!/bin/bash

# needs a command line argument which is path from where you are to the folder
# of interest. eg cxy/n1

for f in $(seq -w 0001 0200); do convert ./$1/0$f.png ./$1/$f.gif ; done
gifsicle --loop --delay 5 --colors 32 -O2 ./$1/*.gif > $1.gif
ffmpeg -r 7 -i ./$1/%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p $1.mp4

rm -v ./$1/*.png
rm -v ./$1/*.gif
