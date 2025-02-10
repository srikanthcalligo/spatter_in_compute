#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0
    uint32_t pgather_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t pgather_coord_x = get_arg_val<uint32_t>(1);
    uint32_t pgather_coord_y =  get_arg_val<uint32_t>(2);
    
    uint32_t pscatter_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t pscatter_coord_x = get_arg_val<uint32_t>(4);
    uint32_t pscatter_coord_y = get_arg_val<uint32_t>(5);

    uint32_t sgather_dram_addr = get_arg_val<uint32_t>(6);
    uint32_t sgather_coord_x = get_arg_val<uint32_t>(7);
    uint32_t sgather_coord_y =  get_arg_val<uint32_t>(8);

    uint32_t sscatter_dram_addr = get_arg_val<uint32_t>(9);
    uint32_t sscatter_coord_x = get_arg_val<uint32_t>(10);
    uint32_t sscatter_coord_y = get_arg_val<uint32_t>(11);
    
    uint32_t n_tiles =  get_arg_val<uint32_t>(12);
    uint32_t pattern_length =  get_arg_val<uint32_t>(13);
    uint32_t delta_gather =  get_arg_val<uint32_t>(14);
    uint32_t delta_scatter =  get_arg_val<uint32_t>(15);
    uint32_t total_count =  get_arg_val<uint32_t>(16);
    
    uint64_t pscatter_dram_noc_addr = get_noc_addr(pscatter_coord_x,pscatter_coord_y,pscatter_dram_addr);
    uint64_t pgather_dram_noc_addr = get_noc_addr(pgather_coord_x,pgather_coord_y,pgather_dram_addr);
    uint64_t sscatter_dram_noc_addr = get_noc_addr(sscatter_coord_x,sscatter_coord_y, sscatter_dram_addr);
    uint64_t sgather_dram_noc_addr = get_noc_addr(sgather_coord_x,sgather_coord_y,sgather_dram_addr);
    
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

    const InterleavedAddrGenFast<true> sscatter_buf = {
        .bank_base_address = sscatter_dram_addr,          // The base address of the buffer
        .page_size = ublock_size_bytes_2,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    const InterleavedAddrGenFast<true> sgather_buf = {
        .bank_base_address = sgather_dram_addr,          // The base address of the buffer
        .page_size = ublock_size_bytes_3,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    // Read data from DRAM -> L1 circular buffers
    noc_async_read(pscatter_dram_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
    noc_async_read_barrier();
    noc_async_read(pgather_dram_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
    noc_async_read_barrier();

    // Do Read operation in RiscV core
    uint32_t* pscatter_data = (uint32_t*) l1_write_addr_in0;
    uint32_t* pgather_data = (uint32_t*) l1_write_addr_in1;
    uint32_t outer_loop_count = 1024 / pattern_length; // 32 * 32

    for(uint32_t tile_id = 0; tile_id < n_tiles; tile_id++) {
        noc_async_read_tile(tile_id, sscatter_buf, l1_write_addr_out0); // read the tile into the circular buffer
        noc_async_read_barrier();

        noc_async_read_tile(tile_id, sgather_buf, l1_write_addr_in2); // read the tile into the circular buffer
        noc_async_read_barrier();

        uint32_t* sscatter_data = (uint32_t*) l1_write_addr_out0;
        uint32_t* sgather_data = (uint32_t*) l1_write_addr_in2;
        for(uint32_t i = 0; i < outer_loop_count ; i++) {
            for(uint32_t j = 0; j < pattern_length; j++) {
                uint32_t index_0 = *(pscatter_data + j) + delta_scatter * i;
                uint32_t index_1 = *(pgather_data + j) + delta_gather * i;
                *(sscatter_data + index_0) = *(sgather_data + index_1);
            }
        }
        // Write data from L1 circulr buffer (out0) -> DRAM
        noc_async_write_tile(tile_id, sscatter_buf, l1_write_addr_out0);
        noc_async_write_barrier();
    }
}