TEMPLATE = subdirs

SUBDIRS = \
  lib \
  test \
  utilities \
  2019 \
  2020

lib.subdir = src/lib
test.subdir = src/test
utilities.subdir = src/utilities
2019.subdir = src/2019
2020.subdir = src/2020

test.depends = lib
utilities.depends = lib
2019.depends = lib
2020.depends = lib
