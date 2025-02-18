#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0

    //Copy data from DRAM to Core0,0 L1
    uint32_t sparse_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_dram_addr = get_arg_val<uint32_t>(1);    
    uint32_t n_tiles =  get_arg_val<uint32_t>(2);    
    
    constexpr uint32_t sparse_cb_id0 = tt::CBIndex::c_0;
    constexpr uint32_t pattern_cb_id1 = tt::CBIndex::c_1;

    uint32_t sparse_tile_size = get_tile_size(sparse_cb_id0);
    uint32_t pattern_tile_size = get_tile_size(pattern_cb_id1);
    
    /*const InterleavedAddrGenFast<true> pattern_arr = {
        .bank_base_address = pattern_dram_addr,
        .page_size = pattern_tile_size,
        .data_format = DataFormat::Float16_b,
    };*/

    ////cb_reserve_back(pattern_cb_id1, 1);

    uint32_t pattern_l1_write_addr_in1 = get_write_ptr(pattern_cb_id1);
    noc_async_read(pattern_dram_addr, pattern_l1_write_addr_in1, pattern_tile_size);
    noc_async_read_barrier();
    
    const InterleavedAddrGenFast<true> sparse_src_buf = {
        .bank_base_address = sparse_dram_addr,          // The base address of the buffer
        .page_size = sparse_tile_size,         // The size of a buffer page
        .data_format = DataFormat::Float16_b, // The data format of the buffer
    };

    for(uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(sparse_cb_id0, 1);
        cb_reserve_back(pattern_cb_id1, 1);

        uint32_t cb_in0_addr = get_write_ptr(sparse_cb_id0);
        noc_async_read_tile(i, sparse_src_buf, cb_in0_addr); // read the tile into the circular buffer
        noc_async_read_barrier();

        cb_push_back(sparse_cb_id0, 1);
        cb_push_back(pattern_cb_id1, 1);
    }
}