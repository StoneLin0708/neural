TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    core/src/nn.cpp \
    load/src/io.cpp \
    load/src/load.cpp \
    load/src/sample.cpp \
    load/src/sampleSet.cpp \
    method/src/algorithm.cpp \
    method/src/method.cpp \
    output/src/info.cpp \
    output/src/plot.cpp \
    load/src/stringProcess.cpp \
    main.cpp \
    core/src/Layer.cpp

HEADERS += \
    core/include/nn.hpp \
    load/include/io.hpp \
    load/include/sample.hpp \
    load/include/sampleSet.hpp \
    method/include/algorithm.hpp \
    method/include/method.hpp \
    output/include/plot.hpp \
    load/include/stringProcess.hpp \
    core/include/Layer.hpp

DISTFILES +=
