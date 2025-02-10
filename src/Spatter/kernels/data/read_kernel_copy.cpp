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
    
    uint32_t noc_addr = get_noc_addr(src0_x,src0_y,dram_addr);
    uint32_t noc_addr1 = get_noc_addr(src1_x,src1_y,dram_addr1);

    constexpr uint32_t cb_id0 = tt::CB::c_in0;
    constexpr uint32_t cb_id1 = tt::CB::c_in1;

    uint32_t tille_size = get_tile_size(cb_id0);
    uint32_t tille_size1 = get_tile_size(cb_id1);
    
    const InterleavedAddrGenFast<true> src0_buf = {
        .bank_base_address = dram_addr,          // The base address of the buffer
        .page_size = tille_size,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };
    
    /*const InterleavedAddrGenFast<true> src1_buf = {
        .bank_base_address = dram_addr1,
        .page_size = tille_size1,
        .data_format = DataFormat::UInt32,
    };
    */
    //for(uint32_t k = 0 ; k < 8 ; k++){
        cb_reserve_back(cb_id1, 1);
        uint32_t l1_addr1 = get_write_ptr(cb_id1);
        noc_async_read(noc_addr1, l1_addr1, tille_size1);
        noc_async_read_barrier();
        uint32_t* data = (uint32_t*)l1_addr1;
        //DPRINT << "On the Device val1: " << F32(data) << ENDL();
        DPRINT << "On the Device val1: " << *data << ENDL();

        cb_push_back(cb_id1, 1);
    //}
    
    for(uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_id0, 1);

        uint32_t cb_in0_addr = get_write_ptr(cb_id0);

        noc_async_read_tile(i, src0_buf, cb_in0_addr); // read the tile into the circular buffer
        //noc_async_read_tile(i, src1_buf, cb_in1_addr); // We can overlap async reads and writes
                                                // to reduce the data movement overhead.

        //noc_async_read(noc_addr, l1_addr0, tille_size);
        //noc_async_read(noc_addr1, l1_addr1, tille_size1);
        noc_async_read_barrier();

        //uint32_t* data = (uint32_t*)l1_addr0;
        //DPRINT << "On the Device val1: " << F32(data) << ENDL();
        //DPRINT << "On the Device val1: " << *data << ENDL();
        cb_push_back(cb_id0, 1);
        //cb_push_back(cb_id1, 1);
    }
    
    
    //noc_async_read_barrier();

    //float data1 = *(float*)l1_addr1;
    //DPRINT << "On the Device val2: " << F32(data1) << ENDL();
    
    //float data2 = data + data1;
    //float data2 = data - data1;
    //float data2 = data * data1;

    //DPRINT << "Riscv engine output(Expected): " << F32(data2) << ENDL();
    //cb_push_back(cb_id1, 1);

}