TEMPLATE = subdirs

SUBDIRS = \
  lib \
  main

lib.subdir = src/lib
main.subdir = src/main

main.depends = lib
