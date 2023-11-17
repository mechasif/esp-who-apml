idf_build_get_property(target IDF_TARGET)

set(srcs            app_main.cpp
                    ../model/custom_coefficient.cpp)

set(src_dirs        ../model
                    .)

set(include_dirs    ../model
                    ../../../../components/esp-dl/include
                    ../../../../components/esp-dl/include/tool
                    ../../../../components/esp-dl/include/typedef
                    ../../../../components/esp-dl/include/nn
                    ../../../../components/esp-dl/include/layer
                    ../../../../components/esp-dl/include/math
                    .)

idf_component_register(SRCS ${srcs} SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

set(lib     libdl.a)

if(${IDF_TARGET} STREQUAL "esp32")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../../components/esp-dl/lib/esp32")

elseif(${IDF_TARGET} STREQUAL "esp32s2")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../../components/esp-dl/lib/esp32s2")

elseif(${IDF_TARGET} STREQUAL "esp32s3")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../../components/esp-dl/lib/esp32s3")

elseif(${IDF_TARGET} STREQUAL "esp32c3")    
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../../components/esp-dl/lib/esp32c3")

endif()


target_link_libraries(${COMPONENT_TARGET} ${links})
target_link_libraries(${COMPONENT_TARGET} ${lib})