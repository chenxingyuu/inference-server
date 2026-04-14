# FindAscendCL.cmake
# Locates Huawei CANN ACL libraries.
# Sets: AscendCL_FOUND, AscendCL::ascendcl

set(_ASCEND_HOME_HINTS
    $ENV{ASCEND_HOME}
    /usr/local/Ascend/ascend-toolkit/latest
    /usr/local/Ascend
)

find_path(AscendCL_INCLUDE_DIR
    NAMES acl/acl.h
    PATHS ${_ASCEND_HOME_HINTS}
    PATH_SUFFIXES acllib/include include
)

find_library(AscendCL_LIBRARY
    NAMES ascendcl
    PATHS ${_ASCEND_HOME_HINTS}
    PATH_SUFFIXES acllib/lib64 lib64
)

find_library(AscendCL_DVPP_LIBRARY
    NAMES acl_dvpp
    PATHS ${_ASCEND_HOME_HINTS}
    PATH_SUFFIXES acllib/lib64 lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AscendCL
    REQUIRED_VARS AscendCL_LIBRARY AscendCL_INCLUDE_DIR)

if(AscendCL_FOUND AND NOT TARGET AscendCL::ascendcl)
    add_library(AscendCL::ascendcl SHARED IMPORTED)
    set_target_properties(AscendCL::ascendcl PROPERTIES
        IMPORTED_LOCATION             "${AscendCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${AscendCL_INCLUDE_DIR}"
    )
    if(AscendCL_DVPP_LIBRARY)
        add_library(AscendCL::dvpp SHARED IMPORTED)
        set_target_properties(AscendCL::dvpp PROPERTIES
            IMPORTED_LOCATION "${AscendCL_DVPP_LIBRARY}"
        )
    endif()
endif()
