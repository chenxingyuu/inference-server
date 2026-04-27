# FindAscendCL.cmake
# Locates Huawei CANN ACL libraries.
# Sets: AscendCL_FOUND, AscendCL::ascendcl
#       CANN_VERSION_MAJOR, CANN_VERSION_MINOR, CANN_VERSION_PATCH

set(_ASCEND_HOME_HINTS
    $ENV{ASCEND_HOME}
    /usr/local/Ascend/ascend-toolkit/latest
    /usr/local/Ascend
)

# ── CANN version detection ─────────────────────────────────────────────────────
set(CANN_VERSION_MAJOR 0 CACHE INTERNAL "")
set(CANN_VERSION_MINOR 0 CACHE INTERNAL "")
set(CANN_VERSION_PATCH 0 CACHE INTERNAL "")

foreach(_hint IN LISTS _ASCEND_HOME_HINTS)
    foreach(_vfile IN ITEMS "version.info" "version.cfg" "ascend_toolkit_install.info")
        if(EXISTS "${_hint}/${_vfile}")
            file(READ "${_hint}/${_vfile}" _vcontent)
            if(_vcontent MATCHES "Version=([0-9]+)\\.([0-9]+)\\.([0-9]+)")
                set(CANN_VERSION_MAJOR "${CMAKE_MATCH_1}" CACHE INTERNAL "" FORCE)
                set(CANN_VERSION_MINOR "${CMAKE_MATCH_2}" CACHE INTERNAL "" FORCE)
                set(CANN_VERSION_PATCH "${CMAKE_MATCH_3}" CACHE INTERNAL "" FORCE)
            endif()
            break()
        endif()
    endforeach()
    if(CANN_VERSION_MAJOR GREATER 0)
        break()
    endif()
endforeach()

find_path(AscendCL_INCLUDE_DIR
    NAMES acl/acl.h
    PATHS ${_ASCEND_HOME_HINTS}
    PATH_SUFFIXES acllib/include aarch64-linux/include x86_64-linux/include include
)

find_library(AscendCL_LIBRARY
    NAMES ascendcl
    PATHS ${_ASCEND_HOME_HINTS}
    PATH_SUFFIXES acllib/lib64 aarch64-linux/lib64 x86_64-linux/lib64 lib64
)

find_library(AscendCL_DVPP_LIBRARY
    NAMES acl_dvpp
    PATHS ${_ASCEND_HOME_HINTS}
    PATH_SUFFIXES acllib/lib64 aarch64-linux/lib64 x86_64-linux/lib64 lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AscendCL
    REQUIRED_VARS AscendCL_LIBRARY AscendCL_INCLUDE_DIR)

if(AscendCL_FOUND)
    message(STATUS "AscendCL: CANN ${CANN_VERSION_MAJOR}.${CANN_VERSION_MINOR}.${CANN_VERSION_PATCH} (from version file)")

    # If version file detection failed, use compile-time feature probing instead.
    # aclDvppVdecCallback was introduced in CANN 7; CANN 6 only has aclvdecCallback.
    if(CANN_VERSION_MAJOR EQUAL 0)
        include(CheckCXXSourceCompiles)
        set(_saved_includes "${CMAKE_REQUIRED_INCLUDES}")
        set(_saved_defs "${CMAKE_REQUIRED_DEFINITIONS}")
        set(CMAKE_REQUIRED_INCLUDES "${AscendCL_INCLUDE_DIR}")
        set(CMAKE_REQUIRED_DEFINITIONS "-DENABLE_DVPP_INTERFACE")
        check_cxx_source_compiles(
            "#include <acl/ops/acl_dvpp.h>\nvoid f(aclDvppVdecCallback){} int main(){}"
            CANN_HAS_ACLDVPP_VDEC_CALLBACK
        )
        set(CMAKE_REQUIRED_INCLUDES "${_saved_includes}")
        set(CMAKE_REQUIRED_DEFINITIONS "${_saved_defs}")

        if(CANN_HAS_ACLDVPP_VDEC_CALLBACK)
            set(CANN_VERSION_MAJOR 8 CACHE INTERNAL "" FORCE)
            message(STATUS "AscendCL: feature probe → CANN 7+ API (aclDvppVdecCallback found)")
        else()
            set(CANN_VERSION_MAJOR 6 CACHE INTERNAL "" FORCE)
            message(STATUS "AscendCL: feature probe → CANN 6 API (aclvdecCallback only)")
        endif()
    endif()
endif()

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
