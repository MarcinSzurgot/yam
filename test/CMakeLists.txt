project(${CMAKE_PROJECT_NAME}-test)
include (FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG release-1.11.0
)

FetchContent_Populate(googletest)
add_subdirectory(
    ${googletest_SOURCE_DIR}
    ${googletest_BINARY_DIR}
)

add_executable(
    ${PROJECT_NAME} 
    
    TestMLPerceptron.cpp
    TestMLTrainer.cpp
    TestUtils.cpp

    test.cpp
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
        gtest
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 23
)