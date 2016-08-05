TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../build-nn-Desktop_Qt_5_7_0_MinGW_32bit-Release/release/ -lnn
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../build-nn-Desktop_Qt_5_7_0_MinGW_32bit-Release/release/ -lnn

INCLUDEPATH += $$PWD/../nn
DEPENDPATH += $$PWD/../nn

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../build-nn-Desktop_Qt_5_7_0_MinGW_32bit-Release/release/libnn.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../build-nn-Desktop_Qt_5_7_0_MinGW_32bit-Release/release/libnn.a

win32: LIBS += -L$$PWD/../../library/armadillo-7.200.2/lib/ -llibarmadillo.dll

INCLUDEPATH += $$PWD/../../library/armadillo-7.200.2/include
DEPENDPATH += $$PWD/../../library/armadillo-7.200.2/include
