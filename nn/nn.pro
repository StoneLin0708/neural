TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    core/src/nn.cpp \
    load/src/sample.cpp \
    method/src/algorithm.cpp \
    method/src/method.cpp \
    output/src/info.cpp \
    output/src/plot.cpp \
    load/src/stringProcess.cpp \
    main.cpp \
    core/src/Layer.cpp \
    load/src/Loader.cpp \
    load/src/sampleFeeder.cpp \
    trainer.cpp

HEADERS += \
    core/include/nn.hpp \
    load/include/sample.hpp \
    method/include/algorithm.hpp \
    method/include/method.hpp \
    output/include/plot.hpp \
    load/include/stringProcess.hpp \
    core/include/Layer.hpp \
    load/include/Loader.hpp \
    load/include/sampleFeeder.hpp \
    trainer.hpp

DISTFILES +=

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../library/armadillo-7.200.2/lib/ -llibarmadillo.dll
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../library/armadillo-7.200.2/lib/ -llibarmadillo.dlld

INCLUDEPATH += $$PWD/../../library/armadillo-7.200.2/include
DEPENDPATH += $$PWD/../../library/armadillo-7.200.2/include
