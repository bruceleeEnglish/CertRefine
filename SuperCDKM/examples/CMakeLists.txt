add_executable(simple_clustering.out simple_clustering.cpp)
target_link_libraries(simple_clustering.out PRIVATE superkmeans)

add_executable(hierarchical_clustering.out hierarchical_clustering.cpp)
target_link_libraries(hierarchical_clustering.out PRIVATE superkmeans)

# SuperCDKM head-to-head benchmark driver
# (the superkmeans INTERFACE target already propagates BLAS_LIBRARIES)
add_executable(supercdkm_bench.out supercdkm_bench.cpp)
target_link_libraries(supercdkm_bench.out PRIVATE superkmeans)

# Custom target to build all examples with 'make examples'
add_custom_target(examples
    DEPENDS
        simple_clustering.out
        hierarchical_clustering.out
        supercdkm_bench.out
)

# go/no-go 实验驱动（支持抽样 / 分层版 / 给定质心初始化）
add_executable(scdkm_exp_bench.out scdkm_exp_bench.cpp)
target_link_libraries(scdkm_exp_bench.out PRIVATE superkmeans)

# CertRefine (C++ port of experiments/refine.py)
add_executable(certrefine.out certrefine.cpp)
target_link_libraries(certrefine.out PRIVATE superkmeans)
