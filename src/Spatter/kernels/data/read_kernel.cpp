#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0

    //Copy data from DRAM to Core0,0 L1
    uint32_t sparse_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t sparse_x_coord = get_arg_val<uint32_t>(2);
    uint32_t sparse_y_coord =  get_arg_val<uint32_t>(3);
    uint32_t pattern_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t pattern_x_coord = get_arg_val<uint32_t>(4);
    uint32_t pattern_y_coord = get_arg_val<uint32_t>(5);
    
    uint32_t n_tiles =  get_arg_val<uint32_t>(6);
    
    //uint32_t dense_dram_addr = get_arg_val<uint32_t>(7);
    //uint32_t dense_x_coord = get_arg_val<uint32_t>(8);
    //uint32_t dense_y_coord = get_arg_val<uint32_t>(9);

    uint32_t pattern_length = get_arg_val<uint32_t>(7);
    uint32_t delta = get_arg_val<uint32_t>(8);
    uint32_t wrap = get_arg_val<uint32_t>(9);

    uint32_t sparse_noc_addr = get_noc_addr(sparse_x_coord,sparse_y_coord,sparse_dram_addr);
    uint32_t pattern_noc_addr = get_noc_addr(pattern_x_coord,pattern_y_coord,pattern_dram_addr);
    //uint32_t dense_noc_addr = get_noc_addr(dense_x_coord, dense_y_coord, dense_dram_addr);

    constexpr uint32_t sparse_cb_id0 = tt::CB::c_in0;
    constexpr uint32_t pattern_cb_id1 = tt::CB::c_in1;
    //constexpr uint32_t dense_cb_id2 = tt::CB::c_in2;

    uint32_t sparse_tile_size = get_tile_size(sparse_cb_id0);
    uint32_t pattern_tile_size = get_tile_size(pattern_cb_id1);
    //uint32_t dense_tile_size = get_tile_size(dense_cb_id2);
    
    
    //uint32_t dense_l1_write_addr_in2 = get_write_ptr(dense_cb_id2);

    const InterleavedAddrGenFast<true> pattern_arr = {
        .bank_base_address = pattern_dram_addr,
        .page_size = pattern_tile_size,
        .data_format = DataFormat::Float16_b,
    };

    cb_reserve_back(pattern_cb_id1, 1);

    uint32_t pattern_l1_write_addr_in1 = get_write_ptr(pattern_cb_id1);
    noc_async_read(pattern_dram_addr, pattern_l1_write_addr_in1, pattern_tile_size);
    //noc_async_read_tile(0, pattern_arr, pattern_l1_write_addr_in1);
    //noc_async_read(dense_dram_addr, dense_l1_write_addr_in2, dense_tile_size);
    noc_async_read_barrier();
    
    // Do Read operation in RiscV core
    //uint32_t* pattern_data = (uint32_t*) pattern_l1_write_addr_in1;
    //uint32_t* dense_data = (uint32_t*) dense_l1_write_addr_in2;
    //float data1 = *(float*)(pattern_l1_write_addr_in1+1);
    //DPRINT << F32(data1) << ENDL();
    const InterleavedAddrGenFast<true> sparse_src_buf = {
        .bank_base_address = sparse_dram_addr,          // The base address of the buffer
        .page_size = sparse_tile_size,         // The size of a buffer page
        .data_format = DataFormat::Float16_b, // The data format of the buffer
    };

    for(uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(sparse_cb_id0, 1);
        uint32_t cb_in0_addr = get_write_ptr(sparse_cb_id0);

        noc_async_read_tile(i, sparse_src_buf, cb_in0_addr); // read the tile into the circular buffer
        noc_async_read_barrier();
        /*
        uint32_t* sparse_data = (uint32_t*) cb_in0_addr;

        for(uint32_t ii = 0; ii < 1024/pattern_length ; ii++) {
            for(uint32_t j = 0; j < pattern_length; j++) {
                uint32_t index_0 = (j + pattern_length * (ii % wrap));
                uint32_t index_1 = (*(pattern_data+j) + delta * ii);
                *(dense_data+index_0) = *(sparse_data+index_1);                
            }
        }*/
        cb_push_back(sparse_cb_id0, 1);
        cb_push_back(pattern_cb_id1, 1);
    }
    //cb_push_back(dense_cb_id2, 1);
}