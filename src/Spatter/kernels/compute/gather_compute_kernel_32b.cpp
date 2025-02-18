// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {

    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 =  tt::CB::c_out0;
    constexpr uint32_t dst_reg = 0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    
    add_tiles_init(cb_in0, cb_in1);
    
    acquire_dst();

    for(uint32_t i = 0; i < n_tiles; i++)
    {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);

        cb_pop_front(cb_in0, 1); 
        cb_pop_front(cb_in1, 1);   
    }
    
    cb_reserve_back(cb_out0, 1);

    pack_tile(dst_reg, cb_out0);
    
    cb_push_back(cb_out0, 1);
    
    release_dst();
}
}