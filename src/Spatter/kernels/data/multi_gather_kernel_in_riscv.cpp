#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0
    uint32_t pgather_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t pgather_coord_x = get_arg_val<uint32_t>(1);
    uint32_t pgather_coord_y =  get_arg_val<uint32_t>(2);
    
    uint32_t pattern_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t pattern_coord_x = get_arg_val<uint32_t>(4);
    uint32_t pattern_coord_y = get_arg_val<uint32_t>(5);

    uint32_t sparse_dram_addr = get_arg_val<uint32_t>(6);
    uint32_t sparse_coord_x = get_arg_val<uint32_t>(7);
    uint32_t sparse_coord_y =  get_arg_val<uint32_t>(8);

    uint32_t dense_dram_addr = get_arg_val<uint32_t>(9);
    uint32_t dense_coord_x = get_arg_val<uint32_t>(10);
    uint32_t dense_coord_y = get_arg_val<uint32_t>(11);
    
    uint32_t n_tiles =  get_arg_val<uint32_t>(12);
    uint32_t pattern_length =  get_arg_val<uint32_t>(13);
    uint32_t delta =  get_arg_val<uint32_t>(14);
    uint32_t wrap =  get_arg_val<uint32_t>(15);
    uint32_t p_len = get_arg_val<uint32_t>(16);
    
    uint64_t pattern_dram_noc_addr = get_noc_addr(pattern_coord_x,pattern_coord_y,pattern_dram_addr);
    uint64_t pgather_dram_noc_addr = get_noc_addr(pgather_coord_x,pgather_coord_y,pgather_dram_addr);
    uint64_t sparse_dram_noc_addr = get_noc_addr(sparse_coord_x,sparse_coord_y, sparse_dram_addr);
    uint64_t dense_dram_noc_addr = get_noc_addr(dense_coord_x,dense_coord_y,dense_dram_addr);
    
    constexpr uint32_t cb_id0 = tt::CB::c_in0;
    constexpr uint32_t cb_id1 = tt::CB::c_in1;
    constexpr uint32_t cb_id2 = tt::CB::c_in2;
    constexpr uint32_t cb_id3 = tt::CB::c_out0;
    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id1);
    uint32_t ublock_size_bytes_2 = get_tile_size(cb_id2);
    uint32_t ublock_size_bytes_3 = get_tile_size(cb_id3);
    
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id0);
    uint32_t l1_write_addr_in1 = get_write_ptr(cb_id1);
    uint32_t l1_write_addr_in2 = get_write_ptr(cb_id2);
    uint32_t l1_write_addr_out0 = get_write_ptr(cb_id3);

    const InterleavedAddrGenFast<true> sparse_buf = {
        .bank_base_address = sparse_dram_addr,          // The base address of the buffer
        .page_size = ublock_size_bytes_2,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    // Read data from DRAM -> L1 circular buffers
    noc_async_read(pattern_dram_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
    noc_async_read_barrier();
    noc_async_read(pgather_dram_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
    noc_async_read_barrier();
    noc_async_read(dense_dram_noc_addr, l1_write_addr_out0, ublock_size_bytes_3);
    noc_async_read_barrier();

    // Do Read operation in RiscV core
    uint32_t* pattern_data = (uint32_t*) l1_write_addr_in0;
    uint32_t* pgather_data = (uint32_t*) l1_write_addr_in1;
    uint32_t* dense_data = (uint32_t*) l1_write_addr_out0;

    uint32_t outer_loop_count = 1024 / p_len; // 32 * 32

    //DPRINT << "DEV " << n_tiles << " " << outer_loop_count << " " << pattern_length << ENDL();
    for(uint32_t tile_id = 0; tile_id < (n_tiles); tile_id++) {
        noc_async_read_tile(tile_id, sparse_buf, l1_write_addr_in2); // read the tile into the circular buffer
        noc_async_read_barrier();
        uint32_t* sparse_data = (uint32_t*) l1_write_addr_in2;
        for(uint32_t i = 0; i < outer_loop_count; i++) {
            for(uint32_t j = 0; j < pattern_length; j++) {
                uint32_t index_0 = j + pattern_length * (i % wrap);
                uint32_t index_1 = *(pattern_data + (*(pgather_data + j))) + delta * i;
                *(dense_data + index_0) = *(sparse_data + index_1);
                //DPRINT << index_0 << " " << index_1 << " " << *(dense_data + index_0) << " "  << *(sparse_data + index_1) << ENDL();
            }
        }
        // Write data from L1 circulr buffer (out0) -> DRAM
        //noc_async_write_tile(tile_id, dense_buf, l1_write_addr_out0);
        //noc_async_write_barrier();
    }
    // Write data from L1 circulr buffer (out0) -> DRAM
    noc_async_write(l1_write_addr_out0, dense_dram_noc_addr, ublock_size_bytes_3);
    noc_async_write_barrier();
}