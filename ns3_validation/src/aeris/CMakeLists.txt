# CMakeLists.txt for AERIS NS-3 module - Complete Validation Suite

# Main AERIS library (header-only for simplicity)
build_lib(
  LIBNAME aeris
  SOURCE_FILES
    model/aeris-protocol.cc
    helper/aeris-helper.cc
  HEADER_FILES
    model/aeris-protocol.h
    model/aeris-protocol-full.h
    model/leach-protocol-ns3.h
    helper/aeris-helper.h
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libmobility}
    ${libenergy}
)

# Simple example
build_lib_example(
  NAME aeris-example
  SOURCE_FILES examples/aeris-example.cc
  LIBRARIES_TO_LINK
    ${libaeris}
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libmobility}
    ${libenergy}
)

# Comprehensive validation example
build_lib_example(
  NAME aeris-validation
  SOURCE_FILES examples/aeris-validation.cc
  LIBRARIES_TO_LINK
    ${libaeris}
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libmobility}
    ${libenergy}
)
