TEMPLATE = app
TARGET = okl

include($$root_dir/config.pri)
include($$root_dir/config_app.pri)

!exists(/usr/include/eigen3/Eigen/Eigen) {
  error("Please install the missing Eigen3 package")
}

INCLUDEPATH += /usr/include/eigen3/Eigen

SOURCES += \
    Params.cpp \
    data/basic_io.cpp \
    main.cpp

HEADERS += \
    Params.h \
    common/ezOptionParser.hpp \
    common/init_param.h \
    common/util.h \
    data/DataPoint.h \
    data/DataReader.h \
    data/DataSet.h \
    data/DataSetHelper.h \
    data/basic_io.h \
    data/comp.h \
    data/io_interface.h \
    data/libsvm_binary.h \
    data/libsvmread.h \
    data/parser.h \
    data/s_array.h \
    data/thread_primitive.h \
    kernel/ik_anne.h \
    kernel/ik_base.h \
    kernel/ik_dot_product.h \
    kernel/ik_iforest.h \
    kernel/ik_mass.h \
    kernel/kernel_RBP.h \
    kernel/kernel_bogd.h \
    kernel/kernel_bpas.h \
    kernel/kernel_fogd.h \
    kernel/kernel_forgetron.h \
    kernel/kernel_ik_ogd.h \
    kernel/kernel_nogd.h \
    kernel/kernel_optim.h \
    kernel/kernel_pa.h \
    kernel/kernel_perceptron.h \
    kernel/kernel_projectron.h \
    kernel/kernel_projectronpp.h \
    kernel/kernel_sgd.h \
    loss/HingeLoss.h \
    loss/LogisticLoss.h \
    loss/LossFunction.h \
    loss/SquareLoss.h \
    loss/SquaredHingeLoss.h
