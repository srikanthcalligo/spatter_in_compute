#include "dataflow_api.h"

void kernel_main(){
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t n_tiles  = get_arg_val<uint32_t>(1);
    uint32_t num_tiles_written = get_arg_val<uint32_t>(2);
    uint32_t num_output_tiles_per_core = get_arg_val<uint32_t>(3);
    uint32_t core_id = get_arg_val<uint32_t>(4);
    uint32_t num_cores = get_arg_val<uint32_t>(5);


    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_2;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);

    const InterleavedAddrGenFast<true> dest = {
        .bank_base_address = dst_addr,
        .page_size = ublock_size_bytes,
        .data_format = DataFormat::Float32,
    };
    //Writing the final tile of each thread to DRAM
    //for(uint32_t tile_id = core_id*num_output_tiles_per_core; tile_id < (core_id*num_output_tiles_per_core+num_output_tiles_per_core); tile_id++) {
        cb_wait_front(cb_id_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_id_out0);
        noc_async_write_tile(core_id, dest, cb_out0_addr); //replace core_id with tile_id to write all tiles
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    //}
}