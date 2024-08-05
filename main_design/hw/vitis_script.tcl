open_project -reset model_top_vitis_hls_project

add_files model.cpp
add_files model.h

add_files -tb tb.cpp
add_files -tb testbench_data

set_top siren_test_model

open_solution "solution1" -flow_target vitis
set_part xczu9eg-ffvb1156-2-e
create_clock -period 3.33 -name default

csynth_design
export_design -format ip_catalog