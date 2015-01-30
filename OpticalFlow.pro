TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    OpticalFlow.cpp \
    opticalflowhs.cpp

include(deployment.pri)
qtcAddDeployment()



HEADERS += \
    OpticalFlow.h \
    opticalflowhs.h \
    common.h


LIBS += -L/usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc

INCLUDEPATH += /usr/local/include

