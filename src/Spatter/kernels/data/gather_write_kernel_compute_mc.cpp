#include "dataflow_api.h"

void kernel_main(){
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_dram_noc_x  = get_arg_val<uint32_t>(1);
    uint32_t dst_dram_noc_y  = get_arg_val<uint32_t>(2);
    uint32_t n_tiles  = get_arg_val<uint32_t>(3);
    uint32_t num_tiles_written = get_arg_val<uint32_t>(4);
    uint32_t num_output_tiles_per_core = get_arg_val<uint32_t>(5);
    uint32_t core_id = get_arg_val<uint32_t>(6);
    uint32_t num_cores = get_arg_val<uint32_t>(7);

    uint64_t dst_noc_addr = get_noc_addr(dst_dram_noc_x, dst_dram_noc_y, dst_addr);

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);

    const InterleavedAddrGenFast<true> dest = {
        .bank_base_address = dst_addr,
        .page_size = ublock_size_bytes,
        .data_format = DataFormat::Float16_b,
    };
    //for(uint32_t i = 0; i < n_tiles; i++) {
    //for(uint32_t tile_id = core_id*num_output_tiles_per_core; tile_id < (core_id*num_output_tiles_per_core+num_output_tiles_per_core); tile_id++) {
    //if(core_id == (num_cores - 1)){
        cb_wait_front(cb_id_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_id_out0);
        noc_async_write_tile(core_id, dest, cb_out0_addr); //tile_id = 0
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    //}
}