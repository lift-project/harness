macro (add_testcase name)
    add_executable (${name} ${name}.cpp)
    target_link_libraries (${name} libgtest)
    add_test (${name} ${name})
endmacro()

add_testcase(test_utils)
#add_testcase(test_matmul)
