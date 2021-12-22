TEMPLATE = lib
TARGET = ml

CONFIG += staticlib create_prl no_install_prl

include($$root_dir/config.pri)

# work around for a broken platform (e.g., Windows)
win32::DEFINES += BRK_WIN

CUDA_SOURCES += \
    math/math_gpu.cu

OTHER_FILES += \
    math/math_gpu.cu

HEADERS += \
    include/mass_ml/config_fs.h \
    include/mass_ml/ds/data_source.h \
    include/mass_ml/ds/ds_file.h \
    include/mass_ml/ds/ds_file_csv.h \
    include/mass_ml/ds/ds_file_libsvm.h \
    include/mass_ml/ds/ds_mdl_dependent.h \
    include/mass_ml/ds/ds_mdl_independent.h \
    include/mass_ml/ds/ds_memory.h \
    include/mass_ml/ds/ds_model.h \
    include/mass_ml/ds/ds_tmp_file.h \
    include/mass_ml/eval/auc.h \
    include/mass_ml/eval/f1_measure.h \
    include/mass_ml/fs/alg_feature_space.h \
    include/mass_ml/fs/alg_fs_aNNE.h \
    include/mass_ml/fs/alg_fs_iNNE.h \
    include/mass_ml/fs/alg_fs_mass.h \
    include/mass_ml/math/math.h \
    include/mass_ml/tr/exception.h \
    include/mass_ml/tr/trace.h \
    include/mass_ml/util/parse_parameter.h

SOURCES += \
    ds/ds_file.cpp \
    ds/ds_file_csv.cpp \
    ds/ds_file_libsvm.cpp \
    ds/ds_mdl_dependent.cpp \
    ds/ds_mdl_independent.cpp \
    ds/ds_memory.cpp \
    ds/ds_tmp_file.cpp \
    eval/auc.cpp \
    eval/f1_measure.cpp \
    fs/alg_fs_aNNE.cpp \
    fs/alg_fs_iNNE.cpp \
    fs/alg_fs_mass.cpp \
    math/math_cpu.cpp \
    tr/exception.cpp
