from fileinput import filename

from PresentRBResults import plot_shots
from srb import mem_qubit_reset, srb_memory

# plot_shots(
#     1,
#     reg_b_copies=1,
#     filename="tests/test2",
# )
plot_shots(
    3,
    mem_err_param=0.40,
    reg_b_copies=3,
    filename="tests/test_with_EC_5",
)
# plot_shots(
#     1,
#     mem_err_param=0.1,
#     reg_b_copies=5,
#     filename="tests/5_copy",
# )
