set(src_core   core/src/network.cpp core/src/layer.cpp)
set(src_layer  layer/src/anfis.cpp layer/src/feedforward.cpp)
set(src_load   load/src/loader.cpp load/src/sample.cpp load/src/samplefeeder.cpp load/src/stringprocess.cpp)
set(src_method method/src/method.cpp method/src/normalize.cpp)
set(src_model  model/anfis_1.cpp model/annmodel.cpp)
set(src_output output/src/info.cpp output/src/plot.cpp)
set(src_train  train/trainer.cpp train/tester.cpp)

set(src ${src_core} ${src_layer} ${src_load} ${src_method} ${src_model} ${src_output} ${src_train})

set(CMAKE_CXX_FLAGS "-o3 -Wall -Wextra -std=c++14")

include_directories(./)

add_library(nn ${src})
