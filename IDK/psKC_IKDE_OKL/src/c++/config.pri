CONFIG -= qt precompile_header
CONFIG += build_all debug_and_release debug_and_release_target
CONFIG += cuda

build_pass:CONFIG(debug, debug|release) {
  TARGET = $$join(TARGET,,,_debug)
}

INCLUDEPATH += $$root_dir/src/lib/include

QMAKE_CXXFLAGS += -m64 -march=native -std=c++17
QMAKE_CXXFLAGS_RELEASE += -DNDEBUG
QMAKE_LFLAGS += -pthread

ide = $$(IDE)

contains(ide, qtcreator) {
  QMAKE_CXXFLAGS += -DML_IDE
}
