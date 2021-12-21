CONFIG -= windows x11
CONFIG += cmdline console link_prl

CONFIG(debug, debug|release) {
  LIBS += -L$$root_dir/build/src/lib/debug

  LIBS += -lml_debug
  PRE_TARGETDEPS += $$root_dir/build/src/lib/debug/libml_debug.a
} else {
  LIBS += -L$$root_dir/build/src/lib/release -lml
  PRE_TARGETDEPS += $$root_dir/build/src/lib/release/libml.a
}

DEPENDPATH += $$root_dir/src/lib/include
