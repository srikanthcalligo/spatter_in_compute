#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0

    //Copy data from DRAM to Core0,0 L1

    uint32_t dram_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_x = get_arg_val<uint32_t>(2);
    uint32_t src0_y =  get_arg_val<uint32_t>(3);
    uint32_t dram_addr1 = get_arg_val<uint32_t>(1);
    uint32_t src1_x = get_arg_val<uint32_t>(4);
    uint32_t src1_y = get_arg_val<uint32_t>(5);
    
    uint32_t n_tiles =  get_arg_val<uint32_t>(6);

    uint32_t dram_addr_out = get_arg_val<uint32_t>(7);
    uint32_t dst_x = get_arg_val<uint32_t>(8);
    uint32_t dst_y =  get_arg_val<uint32_t>(9);

    uint32_t pattern_length =  get_arg_val<uint32_t>(10);
    uint32_t delta =  get_arg_val<uint32_t>(11);
    uint32_t wrap =  get_arg_val<uint32_t>(12);
    
    uint64_t src0_dram_noc_addr = get_noc_addr(src0_x,src0_y,dram_addr);
    uint64_t src1_dram_noc_addr = get_noc_addr(src1_x,src1_y,dram_addr1);
    uint64_t dst_dram_noc_addr = get_noc_addr(dst_x,dst_y,dram_addr_out);
    
    constexpr uint32_t cb_id0 = tt::CB::c_in0;
    constexpr uint32_t cb_id1 = tt::CB::c_in1;
    constexpr uint32_t cb_id2 = tt::CB::c_out0;
    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id1);
    uint32_t ublock_size_bytes_2 = get_tile_size(cb_id2);
    
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id0);
    uint32_t l1_write_addr_in1 = get_write_ptr(cb_id1);
    uint32_t l1_write_addr_out0 = get_write_ptr(cb_id2);

    const InterleavedAddrGenFast<true> src_a_buf = {
        .bank_base_address = dram_addr,          // The base address of the buffer
        .page_size = ublock_size_bytes_0,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    // Read data from DRAM -> L1 circular buffers
    noc_async_read(src1_dram_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
    noc_async_read_barrier();

    // Do Read operation in RiscV core
    uint32_t* dat1 = (uint32_t*) l1_write_addr_in1;
    uint32_t* dat2 = (uint32_t*) l1_write_addr_out0;
    uint32_t outer_loop_count = 1024 / pattern_length; // 32 * 32
    DPRINT << "ublock = " << outer_loop_count << " " << n_tiles << " " << ublock_size_bytes_0  << ENDL();
    //dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];

    for(uint32_t tile_id = 0; tile_id < n_tiles; tile_id++) {
        noc_async_read_tile(tile_id, src_a_buf, l1_write_addr_in0); // read the tile into the circular buffer
        //noc_async_read(src0_dram_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
        noc_async_read_barrier();
        uint32_t* dat0 = (uint32_t*) l1_write_addr_in0;
#if 1
        for(uint32_t i = 0; i < outer_loop_count ; i++) {
            for(uint32_t j = 0; j < pattern_length; j++) {
                //DPRINT << "VAL = " << *(dat0+i) << ENDL();
                //DPRINT << "VAL = " << *(dat1+i) << ENDL();
                //DPRINT << "IN0 = "<< (j + pattern_length * (i % wrap)) << " IN1= " << (*(dat1+j) + delta * i) << ENDL();
                uint32_t index_0 = (j + pattern_length * (i % wrap));
                uint32_t index_1 = (*(dat1+j) + delta * i);
                *(dat2+index_0) = *(dat0+index_1);
                //DPRINT << "IN0 = "<< index_0 << " IN1= " << index_1 << "  "<< *(dat2+index_0) << ENDL();
            }
        }
#endif
    }
    // Write data from L1 circulr buffer (out0) -> DRAM
    noc_async_write(l1_write_addr_out0, dst_dram_noc_addr, ublock_size_bytes_2);
    noc_async_write_barrier();
}