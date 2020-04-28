#!/bin/bash

make all
./run
ffmpeg -framerate 2 -i ./save_folder/Img-%05d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
ffmpeg -i output.mp4 -vf "fps=10,scale=320:-1:flags=lanczos" -c:v pam -f image2pipe - | convert -delay 10 - -loop 0 -layers optimize output.gif
make clean