project(${CMAKE_PROJECT_NAME})

add_executable(
    ${PROJECT_NAME} 
    
    main.cpp
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_PROJECT_NAME}-lib
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 20
)