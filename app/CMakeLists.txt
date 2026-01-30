project(${CMAKE_PROJECT_NAME})

include(FetchContent)

FetchContent_Declare(
    sfml
    GIT_REPOSITORY https://github.com/sfml/sfml
    GIT_TAG 2.5.1
)

FetchContent_GetProperties(sfml)
if (NOT sfml_POPULATED)
    FetchContent_Populate(sfml)
    add_subdirectory(${sfml_SOURCE_DIR} ${sfml_BINARY_DIR})
endif ()

add_executable(
    ${PROJECT_NAME} 
    
    main.cpp
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${sfml_SOURCE_DIR}/include
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        sfml-system
        sfml-window
        sfml-graphics
        pthread
        ${CMAKE_PROJECT_NAME}-lib
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 23
)