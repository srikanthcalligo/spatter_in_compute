#include "dataflow_api.h"

void kernel_main(){
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_dram_noc_x  = get_arg_val<uint32_t>(1);
    uint32_t dst_dram_noc_y  = get_arg_val<uint32_t>(2);
    uint32_t n_tiles  = 1;//get_arg_val<uint32_t>(3);

    uint64_t dst_noc_addr = get_noc_addr(dst_dram_noc_x, dst_dram_noc_y, dst_addr);

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    
    const InterleavedAddrGenFast<true> dest = {
        .bank_base_address = dst_addr,
        .page_size = ublock_size_bytes,
        .data_format = DataFormat::UInt32,
    };
    for(uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_id_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_id_out0);
        noc_async_write_tile(i, dest, cb_out0_addr);
        //noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}