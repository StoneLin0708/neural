#TEMPLATE = lib
#CONFIG += staticlib
TEMPLATE = app
CONFIG += console
CONFIG += c++11
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS_RELEASE -= -O
QMAKE_CXXFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2

QMAKE_CXXFLAGS_RELEASE += -O3

SOURCES += \
    output/src/plot.cpp \
    main.cpp \
    core/src/Layer.cpp \
    load/src/Loader.cpp \
    load/src/Sample.cpp \
    load/src/SampleFeeder.cpp \
    load/src/StringProcess.cpp \
    Trainer.cpp \
    ANNModel.cpp \
    method/src/Normailze.cpp \
    method/src/Method.cpp \
    Tester.cpp \
    output/src/Info.cpp \
    anfis_1.cpp \
    layer/src/anfis.cpp \
    layer/src/feedforward.cpp \
    core/src/network.cpp

HEADERS += \
    output/include/plot.hpp \
    core/include/Layer.hpp \
    load/include/Loader.hpp \
    load/include/SampleFeeder.hpp \
    load/include/Sample.hpp \
    load/include/StringProcess.hpp \
    ANNModel.hpp \
    method/include/Normailze.hpp \
    method/include/Method.hpp \
    Trainer.hpp \
    Tester.hpp \
    Timer.hpp \
    output/include/Info.hpp \
    layer/include/anfis.hpp \
    layer/include/feedforward.hpp \
    core/include/network.hpp

DISTFILES +=

#win32: LIBS += -L$$PWD/../../library/armadillo-7.200.2/lib/ -llibarmadillo.dll

win32:INCLUDEPATH += $$PWD/../../library/armadillo-7.200.2/include
win32:DEPENDPATH += $$PWD/../../library/armadillo-7.200.2/include

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv
