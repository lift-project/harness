set(TEST_SOURCES
    test_utils.cpp)

add_executable (test_all test_all.cpp ${TEST_SOURCES})
target_link_libraries (test_all libgtest Executor ${OpenCL_LIBRARY})
add_test (test_all test_all)
