// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
//#include "debug/dprint.h"
//#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {

    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    
    uint32_t num_tiles_written = get_arg_val<uint32_t>(1);
    uint32_t num_output_tiles_per_core = get_arg_val<uint32_t>(2);
    uint32_t core_id = get_arg_val<uint32_t>(3);
    uint32_t num_cores = get_arg_val<uint32_t>(4);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 =  tt::CBIndex::c_2;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    
    add_tiles_init(cb_in0, cb_in1);

    //DPRINT << "Hello core =" << core_id << ENDL(); 
    
    for(uint32_t tile_id = core_id*num_output_tiles_per_core; tile_id < (core_id*num_output_tiles_per_core+num_output_tiles_per_core); tile_id++)
    {
        //DeviceZoneScopedN("TEST-FULL");
        acquire_dst();
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        add_tiles(cb_in0, cb_in1, 0, 0, 0);

        cb_pop_front(cb_in0, 1); 
        cb_pop_front(cb_in1, 1);
        cb_reserve_back(cb_out0, 1);

        pack_tile(0, cb_out0);
    
        cb_push_back(cb_out0, 1);

        release_dst();   
    }
    
}
}