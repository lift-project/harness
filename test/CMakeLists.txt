
macro (add_testcase name)
    add_executable (${name} ${name}.cpp test_utils.cpp test_matmul.cpp)
    target_link_libraries (${name} libgtest Executor ${OpenCL_LIBRARY})
    add_test (${name} ${name})
endmacro()

add_testcase(test_all)
