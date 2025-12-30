#include "ccv_nnc_mfa_error.hpp"

#include <optional>
#include <vector>

func createMetalSimdgroupEvent() -> String {
  return """
// -*- Metal -*-
//===-- metal_simdgroup_event ---------------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_EVENT
#define __METAL_SIMDGROUP_EVENT

using namespace metal;

enum class simdgroup_async_copy_clamp_mode {
  clamp_to_zero = 0,
  clamp_to_edge = 1
};

// NOTE:
// This is a "no-asm" compatibility implementation.
// It preserves the API surface used by the kernels, but does NOT overlap copy
// latency with compute (it is synchronous).
struct simdgroup_event {
  METAL_FUNC simdgroup_event() thread {}

  template <typename T>
  METAL_FUNC void async_copy(
    threadgroup T *dst,
    const device T *src,
    ulong n_elements
  ) thread {
    for (ulong i = 0; i < n_elements; ++i) {
      dst[i] = src[i];
    }
  }

  template <typename T>
  METAL_FUNC void async_copy(
    device T *dst,
    const threadgroup T *src,
    ulong n_elements
  ) thread {
    for (ulong i = 0; i < n_elements; ++i) {
      dst[i] = src[i];
    }
  }

  template <typename T>
  METAL_FUNC void async_copy(
    // Destination
    threadgroup T *dst,
    ushort dst_elements_per_row,
    ushort2 dst_tile_dimensions,

    // Source
    const device T *src,
    uint src_elements_per_row,
    ushort2 src_tile_dimensions,

    // Other
    bool transpose_matrix = false,
    simdgroup_async_copy_clamp_mode clamp_mode =
      simdgroup_async_copy_clamp_mode::clamp_to_zero
  ) thread {
    // Match the existing behavior: when transpose_matrix is true, we swap the
    // tile dims (we do NOT reorder elements; downstream addressing still uses
    // the transpose flag).
    if (transpose_matrix) {
      src_tile_dimensions = src_tile_dimensions.yx;
      dst_tile_dimensions = dst_tile_dimensions.yx;
    }

    // Copy dst_tile_dimensions; clamp out-of-bounds src reads.
    for (ushort y = 0; y < dst_tile_dimensions.y; ++y) {
      for (ushort x = 0; x < dst_tile_dimensions.x; ++x) {
        bool in_bounds = (x < src_tile_dimensions.x) && (y < src_tile_dimensions.y);

        ulong dst_index = ulong(y) * ulong(dst_elements_per_row) + ulong(x);

        if (in_bounds) {
          ulong src_index = ulong(y) * ulong(src_elements_per_row) + ulong(x);
          dst[dst_index] = src[src_index];
        } else if (clamp_mode == simdgroup_async_copy_clamp_mode::clamp_to_edge &&
                   src_tile_dimensions.x > 0 && src_tile_dimensions.y > 0) {
          ushort sx = min(x, ushort(src_tile_dimensions.x - 1));
          ushort sy = min(y, ushort(src_tile_dimensions.y - 1));
          ulong src_index = ulong(sy) * ulong(src_elements_per_row) + ulong(sx);
          dst[dst_index] = src[src_index];
        } else {
          dst[dst_index] = T(0);
        }
      }
    }
  }

  template <typename T>
  METAL_FUNC void async_copy(
    // Destination
    device T *dst,
    uint dst_elements_per_row,
    ushort2 dst_tile_dimensions,

    // Source
    const threadgroup T *src,
    ushort src_elements_per_row,
    ushort2 src_tile_dimensions,

    // Other
    bool transpose_matrix = false
  ) thread {
    if (transpose_matrix) {
      src_tile_dimensions = src_tile_dimensions.yx;
      dst_tile_dimensions = dst_tile_dimensions.yx;
    }

    // Copy the overlap; avoid OOB if someone passes mismatched tile dims.
    ushort tile_x = min(dst_tile_dimensions.x, src_tile_dimensions.x);
    ushort tile_y = min(dst_tile_dimensions.y, src_tile_dimensions.y);

    for (ushort y = 0; y < tile_y; ++y) {
      for (ushort x = 0; x < tile_x; ++x) {
        ulong dst_index = ulong(y) * ulong(dst_elements_per_row) + ulong(x);
        ulong src_index = ulong(y) * ulong(src_elements_per_row) + ulong(x);
        dst[dst_index] = src[src_index];
      }
    }
  }

  METAL_FUNC static void wait(int /*count*/, thread simdgroup_event* /*events*/) {
    // No-op for synchronous implementation.
  }
};

#endif // __METAL_SIMDGROUP_EVENT
"""
}

std::string createMetalSimdgroupMatrixStorage() {
  // How this header spawning code was designed.
  //
  // Find the patterns between the load/store functions:
  // - device has 'uint' elements_per_row
  // - threadgroup has 'ushort' elements_per_row
  // - both have 'ushort2' matrix_origin
  //
  // The origin is 'ushort2' because the 32-bit part of the address should have
  // been applied previously during 'apply_offset'. The 16-bit part should be
  // hard-coded into the assembly when the GEMM loop is unrolled.
  //
  // Transpose path:
  // - load: reads two values; should split each one onto a separate line.
  //   - overwrites the value of *thread_elements() with a new vec<T, 2>
  // - store: the two instructions are on two separate lines.
  //   - fetches from lane 0 or 1 of thread_elements()[0]
  // - adds 0 or 1 to the hard-coded matrix_origin.x
  //
  // Address generation:
  // - casts some intermediate address fragments to 'ulong' for 'device'
  // - keeps all address fragments in 'ushort' for 'threadgroup'
  
  enum class AddressSpace {
    device,
    threadgroup,
  };
  
  auto keyword =
  [=](AddressSpace value) -> std::string {
    switch (value) {
      case AddressSpace::device:
        return "device";
      case AddressSpace::threadgroup:
        return "threadgroup";
    }
  };
  
  auto offsetType =
  [=](AddressSpace value) -> std::string {
    switch (value) {
      case AddressSpace::device:
        return "uint";
      case AddressSpace::threadgroup:
        return "ushort";
    }
  };
  
  enum class Action {
    load,
    store,
  };
  
  struct MemoryAccessDescriptor {
    std::optional<Action> action;
    std::optional<AddressSpace> addressSpace;
    std::optional<bool> decodingBF16;
    int64_t indentationSpaceCount = 0;
  };
  
  auto createMemoryAccess =
  [=](MemoryAccessDescriptor descriptor) -> std::string {
    CCV_NNC_MFA_PRECONDITION(descriptor.action.has_value());
    CCV_NNC_MFA_PRECONDITION(descriptor.addressSpace.has_value());
    CCV_NNC_MFA_PRECONDITION(descriptor.decodingBF16.has_value());
    auto action = descriptor.action.value();
    auto addressSpace = descriptor.addressSpace.value();
    auto decodingBF16 = descriptor.decodingBF16.value();
    std::string indentation(descriptor.indentationSpaceCount, ' ');
    
    // Determine the arguments.
    std::vector<std::string> arguments;
    auto pointerArgument = [=](std::string dataType) {
      if (action == Action::load) {
        return "const " + keyword(addressSpace) + " " + dataType + " *src";
      } else {
        return keyword(addressSpace) + " " + dataType + " *dst";
      }
    };
    if (decodingBF16) {
      arguments.push_back(pointerArgument("bfloat"));
    } else {
      arguments.push_back(pointerArgument("U"));
    }
    arguments.push_back(offsetType(addressSpace) + " elements_per_row");
    arguments.push_back("ushort2 matrix_origin");
    arguments.push_back("bool transpose_matrix = false");
    
    // Create the warning comment.
    std::string output = "";
    if (decodingBF16) {
      output += indentation + "// WARNING: 'T' must be 'float'.\n";
    } else {
      output += indentation + "template <typename U>\n";
    }
    
    // Create the function signature.
    output += indentation + "METAL_FUNC void";
    if (action == Action::load) {
      output += " load";
    } else {
      output += " store";
    }
    if (decodingBF16) {
      output += "_bfloat";
    }
    output += "(";
    for (int64_t it = 0; it < arguments.size(); ++it) {
      int64_t argumentID = it;
      std::string argument = arguments[argumentID];
      
      output += argument;
      if (argumentID < arguments.size() - 1) {
        output += ", ";
      }
    }
    output += ") {\n";
    
    auto createAddress =
    [=](bool transposed, int64_t offset) -> std::string {
      auto lineY = offsetType(addressSpace) + "(matrix_origin.y)";
      auto lineX = "matrix_origin.x + " + std::to_string(offset);
      lineX = offsetType(addressSpace) + "(" + lineX + ")";
      
      if (transposed) {
        return lineX + " * elements_per_row + " + lineY;
      } else {
        return lineY + " * elements_per_row + " + lineX;
      }
    };
    
    auto createTwoPartAccess =
    [=](bool transposed) -> std::vector<std::string> {
      // Generate the addresses.
      std::vector<std::string> lines;
      for (int64_t laneID = 0; laneID < 2; ++laneID) {
        lines.push_back
        (offsetType(addressSpace) + " address" + std::to_string(laneID) +
         " = " + createAddress(transposed, laneID));
      }
      
      if (action == Action::load) {
        if (decodingBF16) {
          lines.push_back("bfloat memoryForm0 = src[address0]");
          lines.push_back("bfloat memoryForm1 = src[address1]");
        } else {
          lines.push_back("U memoryForm0 = src[address0]");
          lines.push_back("U memoryForm1 = src[address1]");
        }
      }
      
      if (action == Action::load) {
        if (decodingBF16) {
          // Separate the loading logic from the decoding logic for clarity.
          lines.push_back
          ("");
          
          // BF16 decoding logic.
          lines.push_back
          ("bfloat4 registerForm = *(thread bfloat4*)(thread_elements())");
          lines.push_back
          ("registerForm[1] = memoryForm0");
          lines.push_back
          ("registerForm[3] = memoryForm1");
          lines.push_back
          ("((thread bfloat4*)thread_elements())[0] = registerForm");
        } else {
          // Perform a type cast natively supported by the hardware.
          lines.push_back
          ("((thread T*)thread_elements())[0] = T(memoryForm0)");
          lines.push_back
          ("((thread T*)thread_elements())[1] = T(memoryForm1)");
        }
      } else {
        if (decodingBF16) {
          // BF16 encoding logic.
          lines.push_back
          ("bfloat4 registerForm = *(thread bfloat4*)(thread_elements())");
          lines.push_back
          ("registerForm[2] = registerForm[1]");
        } else {
          // Type casts supported natively by the hardware.
          lines.push_back
          ("T registerForm0 = ((thread T*)thread_elements())[0]");
          lines.push_back
          ("T registerForm1 = ((thread T*)thread_elements())[1]");
        }
      }
      
      if (action == Action::store) {
        if (decodingBF16) {
          lines.push_back("dst[address0] = registerForm[2]");
          lines.push_back("dst[address1] = registerForm[3]");
        } else {
          lines.push_back("dst[address0] = U(registerForm0)");
          lines.push_back("dst[address1] = U(registerForm1)");
        }
      }
      return lines;
    };
    
    auto createOnePartAccess =
    [=]() -> std::vector<std::string> {
      std::vector<std::string> lines;
      {
        auto address = createAddress(false, 0);
        lines.push_back("auto combinedAddress = " + address);
      }
      if (action == Action::load) {
        if (decodingBF16) {
          lines.push_back
          ("bfloat2 memoryForm = *(const " +
           keyword(addressSpace) + " packed_bfloat2*)(src + combinedAddress)");
          
          // Separate the loading logic from the decoding logic for clarity.
          lines.push_back
          ("");
          
          // BF16 decoding logic.
          lines.push_back
          ("bfloat4 registerForm = *(thread bfloat4*)(thread_elements())");
          lines.push_back
          ("((thread float*)&registerForm)[1] = *(thread float*)(&memoryForm)");
          lines.push_back
          ("((thread bfloat*)&registerForm)[1] = memoryForm[0]");
          lines.push_back
          ("((thread bfloat4*)thread_elements())[0] = registerForm");
        } else {
          lines.push_back
          ("vec<U, 2> memoryForm = *(const " +
           keyword(addressSpace) + " vec<U, 2>*)(src + combinedAddress)");
          lines.push_back
          ("*(thread_elements()) = vec<T, 2>(memoryForm)");
        }
      } else {
        if (decodingBF16) {
          // BF16 encoding logic.
          lines.push_back
          ("bfloat4 registerForm = *(thread bfloat4*)(thread_elements())");
          lines.push_back
          ("registerForm[2] = registerForm[1]");
          lines.push_back
          ("float memoryForm = ((thread float*)&registerForm)[1]");
          lines.push_back
          ("*(" + keyword(addressSpace) + " float*)" +
           "(dst + combinedAddress) = memoryForm");
        } else {
          lines.push_back
          ("vec<T, 2> registerForm = *(thread_elements())");
          lines.push_back
          ("*(" + keyword(addressSpace) + " vec<U, 2>*)" +
           "(dst + combinedAddress) = vec<U, 2>(registerForm)");
        }
      }
      return lines;
    };
    
    auto insertBlockContents =
    [=](std::vector<std::string>& body, std::vector<std::string> block) {
      for (std::string line : block) {
        // Check whether all characters are whitespace.
        bool allCharactersWhitespace = true;
        for (int8_t character : line) {
          if (isspace(character)) {
            
          } else {
            allCharactersWhitespace = false;
          }
        }
        
        // Branch on the result of this check.
        if (allCharactersWhitespace) {
          body.push_back("  ");
        } else {
          body.push_back("  " + line + ";");
        }
      }
    };
    
    // Determine the lines of the 'if' block.
    std::vector<std::string> body;
    body.push_back("if (transpose_matrix) {");
    insertBlockContents(body, createTwoPartAccess(true));
    
    // Determine the lines of the 'else' block.
    if (decodingBF16) {
      std::vector<std::string> blockContents;
      if (action == Action::load) {
        blockContents = createOnePartAccess();
      } else {
        blockContents = createTwoPartAccess(false);
      }
      
      body.push_back("} else {");
      insertBlockContents(body, blockContents);
      body.push_back("}");
    } else {
      body.push_back("} else if (elements_per_row % 2 != 0) {");
      insertBlockContents(body, createTwoPartAccess(false));
      body.push_back("} else {");
      insertBlockContents(body, createOnePartAccess());
      body.push_back("}");
    }
    
    // Create the function body.
    for (std::string line : body) {
      output += indentation + "  " + line + "\n";
    }
    output += indentation + "}\n";
    return output;
  };
  
  // Add the first section of the shader.
  std::string output;
  output += R"(
// -*- Metal -*-
//===-- metal_simdgroup_matrix_storage ------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_MATRIX_STORAGE
#define __METAL_SIMDGROUP_MATRIX_STORAGE

// The layout of threads within a SIMD matrix.
//
//  0  0  1  1  8  8  9  9
//  2  2  3  3 10 10 11 11
//  4  4  5  5 12 12 13 13
//  6  6  7  7 14 14 15 15
// 16 16 17 17 24 24 25 25
// 18 18 19 19 26 26 27 27
// 20 20 21 21 28 28 29 29
// 22 22 23 23 30 30 31 31
//
// This is Morton order, a method for coalescing data accesses. It is used
// in a variety of contexts, from ray tracing acceleration structures, to
// nodal-point Laplacians, to sorting large lattices of atoms.
//
// Source: https://patents.google.com/patent/US11256518B2
METAL_FUNC static ushort2 morton_order(ushort thread_index_in_simdgroup) {
  ushort lane_id = thread_index_in_simdgroup;
  ushort quad_id = lane_id / 4;
  
  constexpr ushort QUADRANT_SPAN_M = 4;
  constexpr ushort THREADS_PER_QUADRANT = 8;
  ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
  ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
  ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;
  
  ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
  ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
  ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;
  
  return ushort2(N_in_simd, M_in_simd);
}

#pragma METAL internals : enable
namespace metal
{
  template <typename T>
  struct simdgroup_matrix_storage {
    typedef vec<T, 64> storage_type;
    
    storage_type t;
    
    METAL_FUNC thread vec<T, 2>* thread_elements() thread {
      return reinterpret_cast<thread vec<T, 2>*>(&t);
    }
    
    METAL_FUNC simdgroup_matrix_storage() thread = default;
    
    METAL_FUNC simdgroup_matrix_storage(vec<T, 2> thread_elements) thread {
      *(this->thread_elements()) = thread_elements;
    }

    METAL_FUNC static device T* apply_offset(device T *src, uint elements_per_row, uint2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
      } else {
        return src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
      }
    }
    
    METAL_FUNC static threadgroup T* apply_offset(threadgroup T *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + matrix_origin.x * elements_per_row + matrix_origin.y;
      } else {
        return src + matrix_origin.y * elements_per_row + matrix_origin.x;
      }
    }

)";
  
  MemoryAccessDescriptor desc;
  desc.indentationSpaceCount = 4;
  
  std::vector actions = { Action::load, Action::store };
  std::vector addressSpaces = {
    AddressSpace::device, AddressSpace::threadgroup
  };
  std::vector decodingBF16s = { false, true };
  for (auto action : actions) {
    for (auto addressSpace : addressSpaces) {
      for (auto decodingBF16 : decodingBF16s) {
        desc.action = action;
        desc.addressSpace = addressSpace;
        
        desc.decodingBF16 = decodingBF16;
        output += createMemoryAccess(desc);
        output += "\n";
      }
    }
  }
  // Add the last section of the header.
  output += R"(
    template <typename U, typename V>
    METAL_FUNC void multiply(simdgroup_matrix_storage<U> a, simdgroup_matrix_storage<V> b, bool accumulate = true) {
      if (!accumulate) {
        *(thread_elements()) = vec<T, 2>(0);
      }
      t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
    }
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_MATRIX_STORAGE

)";
  return output;
}
