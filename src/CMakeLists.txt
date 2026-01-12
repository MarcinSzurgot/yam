project(${CMAKE_PROJECT_NAME}-lib)

set(INTERFACE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../include/YetAnotherMlp)

add_library(${PROJECT_NAME}
    INTERFACE

    ${INTERFACE_PATH}/Utils.hpp
)

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
        ${INTERFACE_PATH}
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES 
        CXX_STANDARD 23
)