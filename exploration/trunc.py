__author__ = 'Patrick'

import argparse
import os

argparser = argparse.ArgumentParser("Write given number of lines evenly distributed though filename "
                                    "to file called '<file_prefix>_<lines>.<file_extension>'.")
argparser.add_argument("filename")
argparser.add_argument("lines", type=int)
args = argparser.parse_args()

file_prefix, file_extension = os.path.splitext(args.filename)

print file_prefix, file_extension

input_lines = open(args.filename, 'r').readlines()
sample_every = len(input_lines) / args.lines

filtered_lines = [line for (line, index) in zip(input_lines, xrange(len(input_lines))) if index % sample_every == 0]

output_file = open('%s_%s%s' % (file_prefix, args.lines, file_extension), 'w')
for line in filtered_lines[:args.lines]:
    output_file.write(line)

print len(input_lines), sample_every, len(filtered_lines)
