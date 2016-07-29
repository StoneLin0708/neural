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

QMAKE_CXXFLAGS_RELEASE *= -O3

SOURCES += \
    core/src/nn.cpp \
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
    core/src/AnfisLayer.cpp \
    Tester.cpp \
    output/src/Info.cpp \
    anfis_1.cpp

HEADERS += \
    core/include/nn.hpp \
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
    core/include/AnfisLayer.hpp \
    output/include/Info.hpp

DISTFILES +=

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../library/armadillo-7.200.2/lib/ -llibarmadillo.dll
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../library/armadillo-7.200.2/lib/ -llibarmadillo.dll

INCLUDEPATH += $$PWD/../../library/armadillo-7.200.2/include
DEPENDPATH += $$PWD/../../library/armadillo-7.200.2/include

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv
