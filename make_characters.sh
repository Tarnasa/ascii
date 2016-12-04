#!/usr/bin/sh
POINTSIZE=48;

USAGE="$0 [-s <pointsize> -f <font> -c <asciicodes> -o <out_dir>]"

SIZE="12"
FONT="DejaVu-Sans-Mono"
ASCII=`seq 33 126`
OUT="."

while getopts ':s:f:c:o:' opt
do
    case $opt in
        s) SIZE=$OPTARG;;
        f) FONT=$OPTARG;;
        c) ASCII=$OPTARG;;
        o) OUT=$OPTARG;;
       \?) echo "ERROR: Invalid option: $USAGE"
           exit 1;;
    esac
done

for c in $ASCII
do
    STR=`python -c "print('\\\\\\\\' + (chr($c)))"` 
    echo convert \
        -background white \
        -fill black \
        -font $FONT \
        -stroke black \
        -pointsize $POINTSIZE \
        label:$STR "${OUT}/${c}.png"
    convert \
        -background white \
        -fill black \
        -font $FONT \
        -stroke black \
        -pointsize $POINTSIZE \
        label:$STR "${OUT}/${c}.png"
done

