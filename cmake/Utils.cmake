# This is the funtion to use -Wl, --whole-archive to link static library NB:
# target_link_options is broken for this case, it only append the interface link
# options of the first library.
function(kernel_link_options target_name)
  # target_link_options(${target_name} INTERFACE
  # "$<LINK_LIBRARY:WHOLE_ARCHIVE,target_name>")
  target_link_options(
    ${target_name} INTERFACE "SHELL:LINKER:--whole-archive \
    $<TARGET_FILE:${target_name}> \
    LINKER:--no-whole-archive"
  )
endfunction()

# Same as kernel_link_options but it's for MacOS linker
function(macos_kernel_link_options target_name)
  target_link_options(
    ${target_name} INTERFACE
    "SHELL:LINKER:-force_load,$<TARGET_FILE:${target_name}>"
  )
endfunction()

# Ensure that the load-time constructor functions run. By default, the linker
# would remove them since there are no other references to them.
function(target_link_options_shared_lib target_name)
  if(APPLE)
    macos_kernel_link_options(${target_name})
  else()
    kernel_link_options(${target_name})
  endif()
endfunction()

function(download_node_lib version arch target_path)
  # Download node.lib for Windows cross-compilation
  if(WIN32 AND target_path)
    if(NOT EXISTS ${target_path})
      file(DOWNLOAD https://nodejs.org/dist/v${version}/win-${arch}/node.lib ${target_path})
    endif()
  endif()
endfunction()

