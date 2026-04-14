add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -march=native
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g3>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
