// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"


namespace NAMESPACE {
void MAIN {
    //uint32_t per_core_tile_cnt = 1;//get_compile_time_arg_val(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

#if 1
    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 =  tt::CB::c_out0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    
    add_tiles_init();

    acquire_dst(tt::DstMode::Full);
    
    for(uint32_t i = 0; i < n_tiles; i++)
    {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        add_tiles(cb_in0, cb_in1, 0, 0, 0);

        cb_pop_front(cb_in0, 1); 
        cb_pop_front(cb_in1, 1);   
    }

    cb_reserve_back(cb_out0, 1);

    pack_tile(0, cb_out0);
    
    cb_push_back(cb_out0, 1);
    
    release_dst(tt::DstMode::Full);
#endif
#if 0
    //constexpr auto cb_in0 = tt::CB::c_in0;
    //constexpr auto cb_in1 = tt::CB::c_in1;
    //constexpr auto cb_out0 =  tt::CB::c_out0;
    //constexpr auto cb_in0 = tt::CBIndex::c_0;
    //constexpr auto cb_in1 = tt::CBIndex::c_1;
    //constexpr auto cb_out0 = tt::CBIndex::c_16;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in1, tt::CB::c_out0);
    mul_tiles_init();
    //sub_tiles_init();
    //mul_tiles_init();unary_op_init_common
    for(uint32_t i = 0; i < n_tiles; ++i)
    {
        // wait for a block of tiles in each of input CBs
        cb_wait_front(tt::CB::c_in0, 1);
        cb_wait_front(tt::CB::c_in1, 1);

        cb_reserve_back(tt::CB::c_out0, 1);

        tile_regs_acquire(); // acquire 8 tile registers
        //uint32_t aa = 0;
        //copy_tile(tt::CB::c_in0, 0, 0);
        mul_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);
        //mul_tiles(cb_in0, cb_in1, 0, 0, 0);
        /*
        tile_regs_commit(); // signal the packer

        tile_regs_wait(); // packer waits here
        pack_tile(0, tt::CB::c_out0);
        tile_regs_release(); // packer releases
        */
        cb_pop_front(tt::CB::c_in0, 1);
        cb_pop_front(tt::CB::c_in1, 1);
    }
        //cb_reserve_back(tt::CB::c_out0, 1);
        tile_regs_commit(); // signal the packer

        tile_regs_wait(); // packer waits here
        pack_tile(0, tt::CB::c_out0);
        tile_regs_release(); // packer releases
        cb_push_back(tt::CB::c_out0, 1);
    //}
#endif
}
}