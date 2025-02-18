/*!
  \file Configuration.cc
*/

#include <numeric>
#include <atomic>

#include "Configuration.hh"

#include <tt-metalium/host_api.hpp>
//#include "tt_metal/host_api.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/util.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/work_split.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/test_tiles.hpp"
#include "tt-metalium/command_queue.hpp"
//#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include "tt-metalium/tilize_untilize.hpp"
//#include "tt_metal/impl/device/device.hpp"
#include <stdlib.h>


using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

namespace Spatter {

ConfigurationBase::ConfigurationBase(const size_t id, const std::string name,
    std::string k, const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread, double *&dev_dense,
    size_t &dense_size, const size_t delta, const size_t delta_gather,
    const size_t delta_scatter, const long int seed, const size_t wrap,
    const size_t count, const size_t shared_mem, const size_t local_work_size,
    const int nthreads, const unsigned long nruns, const bool aggregate,
    const bool atomic, const unsigned long verbosity)
    : id(id), name(name), kernel(k), pattern(pattern),
      pattern_gather(pattern_gather), pattern_scatter(pattern_scatter),
      sparse(sparse), dev_sparse(dev_sparse), sparse_size(sparse_size),
      sparse_gather(sparse_gather), dev_sparse_gather(dev_sparse_gather),
      sparse_gather_size(sparse_gather_size), sparse_scatter(sparse_scatter),
      dev_sparse_scatter(dev_sparse_scatter),
      sparse_scatter_size(sparse_scatter_size), dense(dense),
      dense_perthread(dense_perthread), dev_dense(dev_dense),
      dense_size(dense_size), delta(delta), delta_gather(delta_gather),
      delta_scatter(delta_scatter), seed(seed), wrap(wrap), count(count),
      shmem(shared_mem), local_work_size(local_work_size),
      omp_threads(nthreads), nruns(nruns), aggregate(aggregate), atomic(atomic),
      verbosity(verbosity), time_seconds(nruns, 0) {
  std::transform(kernel.begin(), kernel.end(), kernel.begin(),
      [](unsigned char c) { return std::tolower(c); });
}

ConfigurationBase::~ConfigurationBase() = default;

int ConfigurationBase::run(bool timed, unsigned long run_id) {
  if (kernel.compare("gather") == 0)
    gather(timed, run_id);
  else if (kernel.compare("scatter") == 0)
    scatter(timed, run_id);
  else if (kernel.compare("sg") == 0)
    scatter_gather(timed, run_id);
  else if (kernel.compare("multigather") == 0)
    multi_gather(timed, run_id);
  else if (kernel.compare("multiscatter") == 0)
    multi_scatter(timed, run_id);
  else {
    std::cerr << "Invalid Kernel Type" << std::endl;
    return -1;
  }

  return 0;
}

void ConfigurationBase::report() {
  size_t bytes_moved = 0;

//Added by Calligo : Changed datatype, we are transferring uint32_t data
#ifdef TT_SPATTER_TIME

  if (kernel.compare("gather") == 0 || kernel.compare("scatter") == 0)
    bytes_moved = pattern.size() * count * sizeof(uint32_t);

  if (kernel.compare("sg") == 0)
    bytes_moved = (pattern_scatter.size() + pattern_gather.size()) * count * sizeof(uint32_t);

  if (kernel.compare("multiscatter") == 0)
    bytes_moved = pattern_scatter.size() * count * sizeof(uint32_t);

  if (kernel.compare("multigather") == 0)
    bytes_moved = pattern_gather.size() * count * sizeof(uint32_t);

#else
  if (kernel.compare("gather") == 0 || kernel.compare("scatter") == 0)
    bytes_moved = pattern.size() * count * sizeof(size_t);

  if (kernel.compare("sg") == 0)
    bytes_moved = (pattern_scatter.size() + pattern_gather.size()) * count * sizeof(size_t);

  if (kernel.compare("multiscatter") == 0)
    bytes_moved = pattern_scatter.size() * count * sizeof(size_t);

  if (kernel.compare("multigather") == 0)
    bytes_moved = pattern_gather.size() * count * sizeof(size_t);
#endif

#ifdef USE_MPI
  int numpes = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numpes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<unsigned long long> vector_bytes_per_run(numpes, 0);
  MPI_Gather(&bytes_moved, 1, MPI_UNSIGNED_LONG_LONG,
      vector_bytes_per_run.data(), 1, MPI_UNSIGNED_LONG_LONG, 0,
      MPI_COMM_WORLD);

  assert(nruns == time_seconds.size());
  std::vector<double> total_time_seconds(nruns, 0.0);
  MPI_Allreduce(time_seconds.data(), total_time_seconds.data(),
      static_cast<int>(nruns), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  long int index = std::distance(total_time_seconds.begin(),
      std::min_element(total_time_seconds.begin(), total_time_seconds.end()));
  assert(index >= 0);
  size_t min_index = static_cast<size_t>(index);

  double mpi_minimum_time = time_seconds[min_index];
  std::vector<double> vector_minimum_time(numpes, 0.0);
  MPI_Gather(&mpi_minimum_time, 1, MPI_DOUBLE, vector_minimum_time.data(), 1,
      MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double mpi_maximum_bandwidth =
      static_cast<double>(bytes_per_run) / mpi_minimum_time / 1000000.0;
  std::vector<double> vector_maximum_bandwidth(numpes, 0.0);
  MPI_Gather(&mpi_maximum_bandwidth, 1, MPI_DOUBLE,
      vector_maximum_bandwidth.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0)
    print_mpi(
        vector_bytes_per_run, vector_minimum_time, vector_maximum_bandwidth);
#else
  double min_time = *std::min_element(time_seconds.begin(), time_seconds.end());  
  double bandwidth = static_cast<double>(bytes_moved) / min_time / 1000000.0;

  print_no_mpi(bytes_moved, min_time, bandwidth);
#endif
}

void ConfigurationBase::setup() {
  if (kernel.compare("multigather") == 0) {
    if (pattern.size() == 0) {
      std::cerr << "Pattern needs to have length of at least 1" << std::endl;
      exit(1);
    }
    if (pattern_gather.size() == 0) {
      std::cerr << "Pattern-Gather needs to have length of at least 1"
                << std::endl;
      exit(1);
    }
  } else if (kernel.compare("multiscatter") == 0) {
    if (pattern.size() == 0) {
      std::cerr << "Pattern needs to have length of at least 1" << std::endl;
      exit(1);
    }
    if (pattern_scatter.size() == 0) {
      std::cerr << "Pattern-Scatter needs to have length of at least 1"
                << std::endl;
      exit(1);
    }
  } else if (kernel.compare("sg") == 0) {
    if (pattern_gather.size() == 0) {
      std::cerr << "Pattern-Gather needs to have length of at least 1"
                << std::endl;
      exit(1);
    }
    if (pattern_scatter.size() == 0) {
      std::cerr << "Pattern-Scatter needs to have length of at least 1"
                << std::endl;
      exit(1);
    }
    if (pattern_scatter.size() != pattern_gather.size()) {
      std::cerr
          << "Pattern-Scatter needs to be the same length as Pattern-gather"
          << std::endl;
      exit(1);
    }
  } else {
    if (pattern.size() == 0) {
      std::cerr << "Pattern needs to have length of at least 1" << std::endl;
      exit(1);
    }
  }

  // Gather and Scatter
  // dense size = pattern.size() * wrap
  // sparse size = max_pattern_val + delta * (count - 1) + 1
  //
  // Concurrent
  // sparse_scatter size = max_pattern_scatter_val + delta_scatter * (count -
  // 1) + 1 sparse_gather size = max_pattern_gather_val + delta_gather *
  // (count - 1) + 1
  //
  // MultiGather
  // dense size = pattern.size() * wrap
  // sparse size = max_pattern_val + delta * (count - 1) + 1
  // assert(pattern.size() > max_pattern_gather_val + 1)
  //
  // MultiScatter
  // dense size = pattern.size() * wrap
  // sparse size = max_pattern_val + delta * (count - 1) + 1
  // assert(pattern.size() > max_pattern_scatter_val + 1)

  if (kernel.compare("sg") == 0) {
    size_t max_pattern_scatter_val = *(std::max_element(
        std::cbegin(pattern_scatter), std::cend(pattern_scatter)));
    size_t max_pattern_gather_val = *(std::max_element(
        std::cbegin(pattern_gather), std::cend(pattern_gather)));
    size_t sparse_scatter_size_ =
        max_pattern_scatter_val + delta_scatter * (count - 1) + 1;
    size_t sparse_gather_size_ =
        max_pattern_gather_val + delta_gather * (count - 1) + 1;

    if (sparse_scatter_size < sparse_scatter_size_)
      sparse_scatter_size = sparse_scatter_size_;

    if (sparse_gather_size < sparse_gather_size_)
      sparse_gather_size = sparse_gather_size_;

    if (verbosity >= 3)
      std::cout << "Pattern Gather Array Size: " << pattern_gather.size()
                << "Pattern Scatter Array Size: " << pattern_scatter.size()
                << "\tDelta: " << delta << "\tCount: " << count
                << "\tWrap: " << wrap
                << "\tSparse Scatter Array Size: " << sparse_scatter_size
                << "\tSparse Gather Array Size: " << sparse_gather_size
                << "\tMax Pattern Scatter Val: " << max_pattern_scatter_val
                << "\tMax Pattern Gather Val: " << max_pattern_gather_val
                << std::endl;
  } else {
    const size_t max_pattern_val =
        *(std::max_element(std::begin(pattern), std::end(pattern)));
    const size_t dense_size_ = pattern.size() * wrap;
    const size_t sparse_size_ = max_pattern_val + delta * (count - 1) + 1;

    if (dense_size < dense_size_)
      dense_size = dense_size_;

    if (sparse_size < sparse_size_)
      sparse_size = sparse_size_;

    if (kernel.compare("multiscatter") == 0) {
      const size_t max_pattern_scatter_val = *(std::max_element(
          std::begin(pattern_scatter), std::end(pattern_scatter)));
      if (pattern.size() <= max_pattern_scatter_val) {
        std::cerr << "Pattern only has length " << pattern.size()
                  << " but needs to have length of at least "
                     "max_pattern_scatter_val = "
                  << max_pattern_scatter_val << std::endl;
        exit(1);
      }
    }

    if (kernel.compare("multigather") == 0) {
      const size_t max_pattern_gather_val = *(std::max_element(
          std::begin(pattern_gather), std::end(pattern_gather)));
      if (pattern.size() <= max_pattern_gather_val) {
        std::cerr << "Pattern only has length " << pattern.size()
                  << " but needs to have length of at least "
                     "max_pattern_gather_val = "
                  << max_pattern_gather_val << std::endl;
        exit(1);
      }
    }

    if (verbosity >= 3) {
      std::cout << "Pattern Array Size: " << pattern.size()
                << "\tDelta: " << delta << "\tCount: " << count
                << "\tWrap: " << wrap << "\tDense Array Size: " << dense_size
                << "\tSparse Array Size: " << sparse_size
                << "\tMax Pattern Val: " << max_pattern_val;

      if (kernel.compare("multiscatter") == 0)
        std::cout << "\tMax Pattern Scatter Val: "
                  << *(std::max_element(std::begin(pattern_scatter),
                         std::end(pattern_scatter)));

      if (kernel.compare("multigather") == 0)
        std::cout << "\tMax Pattern Gather Val: "
                  << *(std::max_element(
                         std::begin(pattern_gather), std::end(pattern_gather)));

      std::cout << std::endl;
    }
  }
}

void ConfigurationBase::print_no_mpi(
    size_t bytes_per_run, double minimum_time, double maximum_bandwidth) {
  std::cout << std::setw(15) << std::left << id << std::setw(15) << std::left
            << bytes_per_run << std::setw(15) << std::left << minimum_time
            << std::setw(15) << std::left << maximum_bandwidth << std::endl;
}

#ifdef USE_MPI
void ConfigurationBase::print_mpi(
    std::vector<unsigned long long> &vector_bytes_per_run,
    std::vector<double> &vector_minimum_time,
    std::vector<double> &vector_maximum_bandwidth) {

  unsigned long long total_bytes = std::accumulate(vector_bytes_per_run.begin(),
      vector_bytes_per_run.end(),
      std::remove_reference_t<decltype(vector_bytes_per_run)>::value_type(0));
  double average_bytes_per_rank = static_cast<double>(total_bytes) /
      static_cast<double>(vector_bytes_per_run.size());

  double total_minimum_time = std::accumulate(vector_minimum_time.begin(),
      vector_minimum_time.end(),
      std::remove_reference_t<decltype(vector_minimum_time)>::value_type(0));
  double average_minimum_time_per_rank =
      total_minimum_time / static_cast<double>(vector_minimum_time.size());

  double total_maximum_bandwidth = std::accumulate(
      vector_maximum_bandwidth.begin(), vector_maximum_bandwidth.end(),
      std::remove_reference_t<decltype(vector_maximum_bandwidth)>::value_type(
          0));
  double average_maximum_bandwidth_per_rank = total_maximum_bandwidth /
      static_cast<double>(vector_maximum_bandwidth.size());

  std::cout << std::setw(15) << std::left << id << std::setw(30) << std::left
            << average_bytes_per_rank << std::setw(30) << std::left
            << total_bytes << std::setw(30) << std::left
            << average_minimum_time_per_rank << std::setw(30) << std::left
            << average_maximum_bandwidth_per_rank << std::setw(30) << std::left
            << total_maximum_bandwidth << std::endl;

  if (verbosity >= 3) {
    std::cout << "\nBytes per rank\n";
    for (unsigned long long bytes : vector_bytes_per_run)
      std::cout << bytes << ' ';
    std::cout << '\n';

    std::cout << "Minimum time per rank(s)\n";
    for (double t : vector_minimum_time)
      std::cout << t << ' ';
    std::cout << '\n';

    std::cout << "Maximum bandwidth per rank(MB/s)\n";
    for (double bw : vector_maximum_bandwidth)
      std::cout << bw << ' ';
    std::cout << std::endl;
  }
}
#endif

std::ostream &operator<<(std::ostream &out, const ConfigurationBase &config) {
  std::stringstream config_output;

  config_output << "{";

  config_output << "'id': " << config.id << ", ";

  if (config.name.compare("") != 0)
    config_output << "'name': '" << config.name << "', ";

  config_output << "'kernel': '" << config.kernel << "', ";

  config_output << "'pattern': [";
  std::copy(std::begin(config.pattern), std::end(config.pattern),
      std::experimental::make_ostream_joiner(config_output, ", "));
  config_output << "], ";

  config_output << "'pattern-gather': [";
  std::copy(std::begin(config.pattern_gather), std::end(config.pattern_gather),
      std::experimental::make_ostream_joiner(config_output, ", "));
  config_output << "], ";

  config_output << "'pattern-scatter': [";
  std::copy(std::begin(config.pattern_scatter),
      std::end(config.pattern_scatter),
      std::experimental::make_ostream_joiner(config_output, ", "));
  config_output << "], ";

  config_output << "'delta': " << config.delta << ", ";
  config_output << "'delta-gather': " << config.delta_gather << ", ";
  config_output << "'delta-scatter': " << config.delta_scatter << ", ";

  config_output << "'count': " << config.count << ", ";

  if (config.seed > 0)
    config_output << "'seed': " << config.seed << ", ";

  if (config.aggregate)
    config_output << "'agg (nruns)': " << config.nruns << ", ";

  config_output << "'wrap': " << config.wrap << ", ";

  config_output << "'threads': " << config.omp_threads;

  config_output << "}";
  return out << config_output.str();
}

Configuration<Spatter::Serial>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread,
    double *&dev_dense, size_t &dense_size,const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const long int seed,
    const size_t wrap, const size_t count, const unsigned long nruns,
    const bool aggregate, const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather,
          delta_scatter, seed, wrap, count, 0, 1024, 1, nruns, aggregate, false,
          verbosity) {
  ConfigurationBase::setup();
}

void Configuration<Spatter::Serial>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();
  int input_run = 0;

  if(strcmp(getenv("TT_SPATTER_FLOAT_VERSION_SERIAL") != NULL ? getenv("TT_SPATTER_FLOAT_VERSION_SERIAL") : "0" , "1") == 0){
    input_run = 1;
  }else if(strcmp(getenv("TT_SPATTER_FLOAT_VERSION_PARALLEL") != NULL ? getenv("TT_SPATTER_FLOAT_VERSION_PARALLEL") : "0" , "1") == 0){
    input_run = 2;
  }else if(strcmp(getenv("TT_SPATTER_BFLOAT16_VERSION_SERIAL") != NULL ? getenv("TT_SPATTER_BFLOAT16_VERSION_SERIAL") : "0" , "1") == 0){
    input_run = 3;
  }else if(strcmp(getenv("TT_SPATTER_BFLOAT16_VERSION_PARALLEL") != NULL ? getenv("TT_SPATTER_BFLOAT16_VERSION_PARALLEL") : "0" , "1") == 0){
    input_run = 4;
  }


#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

//Compute Kernel version
//TT_SPATTER_FLOAT_VERSION_SERIAL
if(input_run == 1) {
    constexpr CoreCoord core = {0,0};
    constexpr uint32_t device_id = 0; 
    IDevice *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();
    uint32_t single_tile_size = 32 * 32;
    uint32_t n_tiles = sparse.size() / single_tile_size;
    uint32_t n_tiles_rem = (sparse.size() % single_tile_size) > 0 ? 1 : 0;
    if(n_tiles_rem > 0){
      n_tiles = n_tiles + 1;
    }
    uint32_t sparse_buf_size = sizeof(float) * single_tile_size * n_tiles;
    uint32_t dense_buf_size = sizeof(float) * single_tile_size;
    uint32_t sparse_buf_page_size = sizeof(float) * single_tile_size;
    uint32_t dense_buf_page_size = sizeof(float) * single_tile_size;

    uint32_t pattern_buf_size = sizeof(float) * single_tile_size;
    uint32_t pattern_buf_page_size = sizeof(float) * single_tile_size;

    printf("Total No.of Tiles = %u Count = %zu\n", n_tiles, count);
    
    tt_metal::BufferConfig buffer_config_sparse = {
            .device = device,
            .size = sparse_buf_size ,
            .page_size = sparse_buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_pattern = {
            .device = device,
            .size = pattern_buf_size,
            .page_size = pattern_buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_dense = {
            .device = device,
            .size = dense_buf_size,
            .page_size = dense_buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse = CreateBuffer(buffer_config_sparse);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern = CreateBuffer(buffer_config_pattern);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_dense = CreateBuffer(buffer_config_dense);

    //auto src0_coord =  dram_buffer_sparse->noc_coordinates();
    //auto src1_coord = dram_buffer_pattern->noc_coordinates();
    //auto dst_coord = dram_buffer_dense->noc_coordinates();

    //Create circular buffer to move data from DRAM to L1
    constexpr uint32_t src0_cb_index = tt::CB::c_in0;

    uint32_t num_tiles_per_cb = 1;
    uint32_t buf_src0 = num_tiles_per_cb * sparse_buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        buf_src0,
        {{src0_cb_index, tt::DataFormat::Float32}}).set_page_size(src0_cb_index, sparse_buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CB::c_in1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        buf_src0,
        {{src1_cb_index, tt::DataFormat::Float32}}).set_page_size(src1_cb_index, pattern_buf_page_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src1_config
    );

    constexpr uint32_t dst_cb_index = tt::CB::c_out0;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        buf_src0,
        {{dst_cb_index, tt::DataFormat::Float32}}).set_page_size(dst_cb_index, dense_buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb_dst_config
    );

    //Create datamovement kernels
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/gather_read_kernel_compute_32b.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/gather_write_kernel_compute_32b.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/compute/gather_compute_kernel_32b.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
    //Declare pattern and sparse arrays
    std::vector<uint32_t> sparse_vec(single_tile_size * n_tiles);
    std::vector<uint32_t> pattern_mat_val(single_tile_size);
    
    for(uint32_t i=0; i < sparse.size() ; i++){
      sparse_vec[i] = (float)sparse[i];
    }
    uint32_t stride_len = pattern[1];
    float in_val = 0;

    for(int i=0; i < single_tile_size ; i++){
      pattern_mat_val[i] = (uint32_t)in_val;
    }

    EnqueueWriteBuffer(cq, dram_buffer_sparse, sparse_vec, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern, pattern_mat_val, false);

    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer_sparse->address(), dram_buffer_pattern->address(), n_tiles});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer_dense->address(), 1}); //Return only the final tile

    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }

    std::vector<uint32_t> dense_vec(single_tile_size);
    EnqueueReadBuffer(cq, dram_buffer_dense, dense_vec, true);

#ifdef PRINT_DEBUG
    printf("Dense_size = %zu pattern_size = %zu sparse_size = %zu\n",dense_vec.size(), pattern_mat_val.size(), sparse_vec.size());
    printf("TT Input : \n");
    for(uint32_t i= sparse.size() - (pattern_length * stride_len); i < sparse.size(); i = i+stride_len){
      printf("%f ", (float)sparse_vec[i]);
    }
    printf("\n");
    printf("TT Result : \n");
    if(n_tiles_rem > 0){
      for(uint32_t i= 0; i < sparse.size() % single_tile_size; i = i+stride_len){
        printf("%f ", (float)dense_vec[i]);
      }
    }
    else {
      for(uint32_t i= (dense_vec.size()) - pattern_length; i < dense_vec.size(); i = i+stride_len){
        printf("%f ", (float)dense_vec[i]);
      }
    }
    printf("\n\n");

    printf("Host Result : \n");

    for (size_t i = 0; i < count; ++i){
    for (size_t j = 0; j < pattern_length; ++j){
      dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
      if(i == count - 1){
        printf("%f ", dense[j + pattern_length * (i % wrap)]);
      }
    }
  }
#endif
    CloseDevice(device);

}else if(input_run == 2) { //TT_SPATTER_FLOAT_VERSION_PARALLEL
    constexpr uint32_t device_id = 0; 
    IDevice *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    /*
    * Multi-Core prep
    */
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    //auto compute_with_storage_grid_size = device->grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    constexpr CoreCoord core = {0,0};
    uint32_t single_tile_size = 32 * 32;
    uint32_t n_tiles = (count * pattern_length)/ single_tile_size; //1024 = 32 * 32
    uint32_t n_tiles_rem = (sparse.size() % single_tile_size) > 0 ? 1 : 0;
    if(n_tiles_rem > 0){
      n_tiles = n_tiles + 1;
    }
    uint32_t buf_size = sizeof(float) * single_tile_size * n_tiles;
    
    auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);

    uint32_t dest_buf_size = sizeof(float) * num_cores * single_tile_size; // num_cores
    uint32_t buf_page_size = sizeof(float) * single_tile_size;
    uint32_t pattern_buf_size = sizeof(float) * single_tile_size;

    std::cout << "No.of Tiles = " << n_tiles << "  No.of Cores = " << num_cores << std::endl;

    //std::cout << core_group_1.num_cores() << std::endl;
    //std::cout << core_group_2.num_cores() << std::endl;
    //std::cout << num_output_tiles_per_core_group_1 << std::endl;
    //std::cout << num_output_tiles_per_core_group_2 << std::endl;

    tt_metal::BufferConfig buffer_config_a = {
            .device = device,
            .size = buf_size ,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_b = {
            .device = device,
            .size = pattern_buf_size,
            .page_size = pattern_buf_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_c = {
            .device = device,
            .size = dest_buf_size,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer1 = CreateBuffer(buffer_config_b);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer2 = CreateBuffer(buffer_config_c);

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;

    uint32_t num_tiles_per_cb = 1;
    uint32_t buf_src0 = num_tiles_per_cb * buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src0_cb_index, tt::DataFormat::Float32}}).set_page_size(src0_cb_index, buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        num_tiles_per_cb * pattern_buf_size,
        {{src1_cb_index, tt::DataFormat::Float32}}).set_page_size(src1_cb_index, pattern_buf_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb1_src1_config
    );

    constexpr uint32_t dst_cb_index = tt::CBIndex::c_2;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        num_tiles_per_cb * dest_buf_size,
        {{dst_cb_index, tt::DataFormat::Float32}}).set_page_size(dst_cb_index, buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb_dst_config
    );

    //Create datamovement kernels
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/gather_read_kernel_compute_mc_32b.cpp",
                    all_cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/gather_write_kernel_compute_mc_32b.cpp",
                    all_cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/compute/gather_compute_kernel_mc.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
    //Input pattern and sparse arrary
    std::vector<uint32_t> sparse_vec(n_tiles * single_tile_size);
    std::vector<uint32_t> pattern_mat_val(single_tile_size);
    
    for(uint32_t i=0; i < sparse.size() ; i++){
      sparse_vec[i] =  (float)sparse[i];
    }

    uint32_t stride_len = pattern[1];
    float in_val = 0;

    for(int i=0; i < single_tile_size ; i++){
      pattern_mat_val[i] = (uint32_t)in_val;
    }
    EnqueueWriteBuffer(cq, dram_buffer, sparse_vec, false);
    EnqueueWriteBuffer(cq, dram_buffer1, pattern_mat_val, false);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
 
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
        num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        SetRuntimeArgs(program,
         void_data_kernel_noc0_read,
         core,
         {dram_buffer->address(),
         dram_buffer1->address(),
         n_tiles,
         num_tiles_written,
         num_output_tiles_per_core, 
         i});

        SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
        SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer2->address(), n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
        num_tiles_written += num_output_tiles_per_core;
    }

    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
    //Output data
    std::vector<uint32_t> dense_vec(single_tile_size * num_cores);
    EnqueueReadBuffer(cq, dram_buffer2, dense_vec, true);

#ifdef PRINT_DEBUG  
    printf("Dense_size = %zu pattern_size = %zu sparse_size = %zu\n",dense_vec.size(), pattern_mat_val.size(), sparse_vec.size());
    
    printf("TT Input : \n");
    for(uint32_t i= sparse.size() - (pattern_length * stride_len); i < sparse.size(); i = i+stride_len){
      printf("%f ", (float)sparse_vec[i]);
    }

    printf("\n");
    printf("TT Result : \n");
    //if(n_tiles_rem > 0){
    //  for(uint32_t i= (dense_vec.size() * 2) - 512; i < dense_vec.size() * 2 - 20; i = i+stride_len){
    //    printf("%f ", c_bf16[i].to_float());
    //  }
    // }
    //else {
      for(uint32_t i= dense_vec.size() - pattern_length; i < dense_vec.size(); i = i+stride_len){
        printf("%f ", (float)dense_vec[i]);
      }  
    //}
    printf("\n\n");
#endif
    CloseDevice(device);


}else if(input_run == 3){ //TT_SPATTER_BFLOAT16_VERSION_SERIAL
    constexpr CoreCoord core = {0,0};
    constexpr uint32_t device_id = 0; 
    IDevice *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();
    uint32_t single_tile_size = 32 * 32;
    uint32_t n_tiles = sparse.size() / single_tile_size;
    uint32_t n_tiles_rem = (sparse.size() % single_tile_size) > 0 ? 1 : 0;
    if(n_tiles_rem > 0){
      n_tiles = n_tiles + 1;
    }
    uint32_t sparse_buf_size = sizeof(bfloat16) * single_tile_size * n_tiles;
    uint32_t dense_buf_size = sizeof(bfloat16) * single_tile_size;
    uint32_t sparse_buf_page_size = sizeof(bfloat16) * single_tile_size;
    uint32_t dense_buf_page_size = sizeof(bfloat16) * single_tile_size;

    uint32_t pattern_buf_size = sizeof(bfloat16) * single_tile_size;
    uint32_t pattern_buf_page_size = sizeof(bfloat16) * single_tile_size;

    printf("Total No.of Tiles = %u Count = %zu\n", n_tiles, count);
    
    tt_metal::BufferConfig buffer_config_sparse = {
            .device = device,
            .size = sparse_buf_size ,
            .page_size = sparse_buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_pattern = {
            .device = device,
            .size = pattern_buf_size,
            .page_size = pattern_buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_dense = {
            .device = device,
            .size = dense_buf_size,
            .page_size = dense_buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse = CreateBuffer(buffer_config_sparse);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern = CreateBuffer(buffer_config_pattern);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_dense = CreateBuffer(buffer_config_dense);

    //Create circular buffer to move data from DRAM to L1
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;

    uint32_t num_tiles_per_cb = 1;
    uint32_t buf_src0 = num_tiles_per_cb * sparse_buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        buf_src0,
        {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, sparse_buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        buf_src0,
        {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, pattern_buf_page_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src1_config
    );

    constexpr uint32_t dst_cb_index = tt::CBIndex::c_2;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        buf_src0,
        {{dst_cb_index, tt::DataFormat::Float16_b}}).set_page_size(dst_cb_index, dense_buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb_dst_config
    );

    //Create datamovement kernels
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/gather_read_kernel_compute.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/gather_write_kernel_compute.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/compute/gather_compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
    //Declare pattern and sparse arrays
    std::vector<bfloat16> sparse_vec(single_tile_size * n_tiles);
    std::vector<bfloat16> pattern_mat_val(single_tile_size);

    for(uint32_t i=0; i < sparse.size() ; i++){
      sparse_vec[i] = bfloat16((float)sparse[i]);
    }
    uint32_t stride_len = pattern[1];
    float in_val = 0;
    for(int i=0; i < single_tile_size ; i++){
      pattern_mat_val[i] = bfloat16(in_val);
    }

    EnqueueWriteBuffer(cq, dram_buffer_sparse, sparse_vec, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern, pattern_mat_val, false);

    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer_sparse->address(), dram_buffer_pattern->address(), n_tiles});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer_dense->address(), 1}); //Return only the final tile

    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }

    std::vector<bfloat16> dense_vec(single_tile_size);
    EnqueueReadBuffer(cq, dram_buffer_dense, dense_vec, true);

    CloseDevice(device);

#ifdef PRINT_DEBUG
    printf("Dense_size = %zu pattern_size = %zu sparse_size = %zu\n",dense_vec.size(), pattern_mat_val.size(), sparse_vec.size());
    printf("TT Input : \n");
    for(uint32_t i= sparse.size() - (pattern_length * stride_len); i < sparse.size(); i = i+stride_len){
      printf("%f ", sparse_vec[i].to_float());
    }
    printf("\n\n");
    printf("TT Result : \n");
    if(n_tiles_rem > 0){
      for(uint32_t i= 0; i < sparse.size() % single_tile_size; i = i+stride_len){
        printf("%f ", dense_vec[i].to_float());
      }
    }
    else {
      for(uint32_t i= dense_vec.size() - pattern_length; i < dense_vec.size(); i = i+stride_len){
        printf("%f ", dense_vec[i].to_float());
      }
    }
    printf("\n\n");
    printf("Host Result : \n");
    for (size_t i = 0; i < count; ++i){
    for (size_t j = 0; j < pattern_length; ++j){
      dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
      if(i == count - 1){
        printf("%f ", dense[j + pattern_length * (i % wrap)]);
      }
    }
  }
#endif

}else if(input_run == 4){ //TT_SPATTER_BFLOAT16_VERSION_PARALLEL
    constexpr uint32_t device_id = 0; 
    IDevice *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    /*
    * Multi-Core prep
    */
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    //auto compute_with_storage_grid_size = device->grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    constexpr CoreCoord core = {0,0};
    uint32_t single_tile_size = 32 * 32;
    uint32_t n_tiles = (count * pattern_length)/ single_tile_size; //1024 = 32 * 32
    uint32_t n_tiles_rem = (sparse.size() % single_tile_size) > 0 ? 1 : 0;
    if(n_tiles_rem > 0){
      n_tiles = n_tiles + 1;
    }
    uint32_t buf_size = sizeof(bfloat16) * single_tile_size * n_tiles;
    
    auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);

    uint32_t dest_buf_size = sizeof(bfloat16) * num_cores * single_tile_size; // num_cores
    uint32_t buf_page_size = sizeof(bfloat16) * single_tile_size;
    uint32_t pattern_buf_size = sizeof(bfloat16) * single_tile_size;

    std::cout << "No.of Tiles = " << n_tiles << "  No.of Cores = " << num_cores << std::endl;

    //std::cout << core_group_1.num_cores() << std::endl;
    //std::cout << core_group_2.num_cores() << std::endl;
    //std::cout << num_output_tiles_per_core_group_1 << std::endl;
    //std::cout << num_output_tiles_per_core_group_2 << std::endl;

    tt_metal::BufferConfig buffer_config_a = {
            .device = device,
            .size = buf_size ,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_b = {
            .device = device,
            .size = pattern_buf_size,
            .page_size = pattern_buf_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_c = {
            .device = device,
            .size = dest_buf_size,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer1 = CreateBuffer(buffer_config_b);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer2 = CreateBuffer(buffer_config_c);

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;

    uint32_t num_tiles_per_cb = 1;
    uint32_t buf_src0 = num_tiles_per_cb * buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        num_tiles_per_cb * pattern_buf_size,
        {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, pattern_buf_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb1_src1_config
    );

    constexpr uint32_t dst_cb_index = tt::CBIndex::c_2;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        num_tiles_per_cb * dest_buf_size,
        {{dst_cb_index, tt::DataFormat::Float16_b}}).set_page_size(dst_cb_index, buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb_dst_config
    );

    //Create datamovement kernels
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/gather_read_kernel_compute_mc.cpp",
                    all_cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/gather_write_kernel_compute_mc.cpp",
                    all_cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/compute/gather_compute_kernel_mc.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
    //Input pattern and sparse arrary
    std::vector<bfloat16> sparse_vec(n_tiles * single_tile_size);
    std::vector<bfloat16> pattern_mat_val(single_tile_size);
    
    for(uint32_t i=0; i < sparse.size() ; i++){
      sparse_vec[i] =  bfloat16((float)sparse[i]);
    }

    uint32_t stride_len = pattern[1];
    float in_val = 0;

    for(int i=0; i < single_tile_size ; i++){
      pattern_mat_val[i] = bfloat16(in_val);
    }
    EnqueueWriteBuffer(cq, dram_buffer, sparse_vec, false);
    EnqueueWriteBuffer(cq, dram_buffer1, pattern_mat_val, false);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
 
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
        num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        SetRuntimeArgs(program,
         void_data_kernel_noc0_read,
         core,
         {dram_buffer->address(),
         dram_buffer1->address(),
         n_tiles,
         num_tiles_written,
         num_output_tiles_per_core, 
         i});

        SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
        SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer2->address(), n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
        num_tiles_written += num_output_tiles_per_core;
    }

    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }

    //Output data
    std::vector<bfloat16> dense_vec(single_tile_size * num_cores);
    EnqueueReadBuffer(cq, dram_buffer2, dense_vec, true);

#ifdef PRINT_DEBUG  
    printf("Dense_size = %zu pattern_size = %zu sparse_size = %zu\n",dense_vec.size(), pattern_mat_val.size(), sparse_vec.size());
    
    printf("TT Input : \n");
    for(uint32_t i= sparse.size() - (pattern_length * stride_len); i < sparse.size(); i = i+stride_len){
      printf("%f ", sparse_vec[i].to_float());
    }

    printf("\n");
    printf("TT Result : \n");
    //if(n_tiles_rem > 0){
    //  for(uint32_t i= (dense_vec.size() * 2) - 512; i < dense_vec.size() * 2 - 20; i = i+stride_len){
    //    printf("%f ", c_bf16[i].to_float());
    //  }
    // }
    //else {
      for(uint32_t i= dense_vec.size() - (pattern_length * stride_len); i < dense_vec.size(); i = i+stride_len){
        printf("%f ", dense_vec[i].to_float());
      }  
    //}
    printf("\n\n");
#endif
    CloseDevice(device);

//Host Code
}else {

  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i){
    for (size_t j = 0; j < pattern_length; ++j){
      dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
    }
  }
  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }

}
}

void Configuration<Spatter::Serial>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

  int input_run = 0;
  if(strcmp(getenv("TT_SPATTER_BFLOAT16_VERSION_SERIAL") != NULL ? getenv("TT_SPATTER_BFLOAT16_VERSION_SERIAL") : "0" , "1") == 0){
    input_run = 1;
  }else if(strcmp(getenv("TT_SPATTER_BFLOAT16_VERSION_PARALLEL") != NULL ? getenv("TT_SPATTER_BFLOAT16_VERSION_PARALLEL") : "0" , "1") == 0){
    input_run = 2;
  }

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
    
if(input_run == 1) { //TT_SPATTER_BFLOAT16_VERSION_SERIAL
    constexpr CoreCoord core = {0,0};
    constexpr uint32_t device_id = 0; 
    IDevice *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();
    uint32_t single_tile_size = 32 * 32;
    uint32_t n_tiles = sparse.size() / single_tile_size;
    uint32_t n_tiles_rem = (sparse.size() % single_tile_size) > 0 ? 1 : 0;
    if(n_tiles_rem > 0){
      n_tiles = n_tiles + 1;
    }
    uint32_t sparse_buf_size = sizeof(bfloat16) * single_tile_size * n_tiles;
    uint32_t dense_buf_size = sizeof(bfloat16) * single_tile_size;
    uint32_t sparse_buf_page_size = sizeof(bfloat16) * single_tile_size;
    uint32_t dense_buf_page_size = sizeof(bfloat16) * single_tile_size;

    uint32_t pattern_buf_size = sizeof(bfloat16) * single_tile_size;
    uint32_t pattern_buf_page_size = sizeof(bfloat16) * single_tile_size;

    printf("Total No.of Tiles = %u Count = %zu\n", n_tiles, count);
    
    tt_metal::BufferConfig buffer_config_sparse = {
            .device = device,
            .size = sparse_buf_size ,
            .page_size = sparse_buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_pattern = {
            .device = device,
            .size = pattern_buf_size,
            .page_size = pattern_buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_dense = {
            .device = device,
            .size = dense_buf_size,
            .page_size = dense_buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse = CreateBuffer(buffer_config_sparse);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern = CreateBuffer(buffer_config_pattern);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_dense = CreateBuffer(buffer_config_dense);

    //Create circular buffer to move data from DRAM to L1
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;

    uint32_t num_tiles_per_cb = 1;
    uint32_t buf_src0 = num_tiles_per_cb * dense_buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        buf_src0,
        {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, dense_buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        buf_src0,
        {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, pattern_buf_page_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src1_config
    );

    constexpr uint32_t dst_cb_index = tt::CBIndex::c_2;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        buf_src0,
        {{dst_cb_index, tt::DataFormat::Float16_b}}).set_page_size(dst_cb_index, sparse_buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb_dst_config
    );

    //Create datamovement kernels
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/scatter_read_kernel_compute.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/scatter_write_kernel_compute.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/compute/scatter_compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
    //Declare pattern and sparse arrays
    std::vector<bfloat16> dense_vec(single_tile_size);
    std::vector<bfloat16> pattern_mat_val(single_tile_size);

    for(uint32_t i=0; i < single_tile_size ; i++){
        dense_vec[i] = bfloat16((float)dense[i % pattern_length]);
    }
    uint32_t stride_len = pattern[1];
    float in_val = 0;
    for(int i=0; i < single_tile_size ; i++){
      pattern_mat_val[i] = bfloat16(in_val);
    }

    EnqueueWriteBuffer(cq, dram_buffer_dense, dense_vec, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern, pattern_mat_val, false);

    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer_dense->address(), dram_buffer_pattern->address(), n_tiles});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer_sparse->address(), n_tiles});

    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);
    
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
    std::vector<bfloat16> sparse_vec;

    EnqueueReadBuffer(cq, dram_buffer_sparse, sparse_vec, true);
    
    CloseDevice(device);

#ifdef PRINT_DEBUG
    printf("Dense_size = %zu pattern_size = %zu sparse_size = %zu\n",dense_vec.size(), pattern_mat_val.size(), sparse_vec.size());
    printf("TT Input : \n");
    for(uint32_t i= 0; i < dense.size(); i++){
      printf("%f ", dense_vec[i].to_float());
    }
    printf("\n\n");
    printf("TT Result : \n");
    
    for(uint32_t i= sparse_vec.size() - (pattern_length) ; i < sparse_vec.size(); i++){
        printf("%f ", sparse_vec[i].to_float());
    }
      
    printf("\n");
#endif

} else if(input_run == 2){ //TT_SPATTER_BFLOAT16_VERSION_PARALLEL
    constexpr uint32_t device_id = 0; 
    IDevice *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    /*
    * Multi-Core prep
    */
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    //auto compute_with_storage_grid_size = device->grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    constexpr CoreCoord core = {0,0};
    uint32_t single_tile_size = 32 * 32;
    uint32_t n_tiles = (count * pattern_length)/ single_tile_size; //1024 = 32 * 32
    uint32_t n_tiles_rem = (sparse.size() % single_tile_size) > 0 ? 1 : 0;
    if(n_tiles_rem > 0){
      n_tiles = n_tiles + 1;
    }
    uint32_t buf_size = sizeof(bfloat16) * single_tile_size * n_tiles;
    
    auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);

    uint32_t dest_buf_size = sizeof(bfloat16) * single_tile_size; // num_cores
    uint32_t buf_page_size = sizeof(bfloat16) * single_tile_size;
    uint32_t pattern_buf_size = sizeof(bfloat16) * single_tile_size;

    std::cout << "No.of Tiles = " << n_tiles << "  No.of Cores = " << num_cores << std::endl;

    //std::cout << core_group_1.num_cores() << std::endl;
    //std::cout << core_group_2.num_cores() << std::endl;
    //std::cout << num_output_tiles_per_core_group_1 << std::endl;
    //std::cout << num_output_tiles_per_core_group_2 << std::endl;

    tt_metal::BufferConfig buffer_config_a = {
            .device = device,
            .size = buf_size ,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_b = {
            .device = device,
            .size = pattern_buf_size,
            .page_size = pattern_buf_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_c = {
            .device = device,
            .size = dest_buf_size,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer1 = CreateBuffer(buffer_config_b);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer2 = CreateBuffer(buffer_config_c);

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;

    uint32_t num_tiles_per_cb = 1;
    uint32_t buf_src0 = num_tiles_per_cb * buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        num_tiles_per_cb * pattern_buf_size,
        {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, pattern_buf_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb1_src1_config
    );

    constexpr uint32_t dst_cb_index = tt::CBIndex::c_2;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        num_tiles_per_cb * dest_buf_size,
        {{dst_cb_index, tt::DataFormat::Float16_b}}).set_page_size(dst_cb_index, buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb_dst_config
    );

    //Create datamovement kernels
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/scatter_read_kernel_compute_mc.cpp",
                    all_cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/data/scatter_write_kernel_compute_mc.cpp",
                    all_cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter_in_compute/src/Spatter/kernels/compute/scatter_compute_kernel_mc.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
    //Input pattern and sparse arrary
    std::vector<bfloat16> dense_vec(single_tile_size);
    std::vector<bfloat16> pattern_mat_val(single_tile_size);
    
    for(uint32_t i=0; i < single_tile_size ; i++){
      dense_vec[i] =  bfloat16((float)dense[i % pattern_length]);
    }

    uint32_t stride_len = pattern[1];
    float in_val = 0;

    for(int i=0; i < single_tile_size ; i++){
      pattern_mat_val[i] = bfloat16(in_val);
    }
    EnqueueWriteBuffer(cq, dram_buffer2, dense_vec, false);
    EnqueueWriteBuffer(cq, dram_buffer1, pattern_mat_val, false);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
 
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
        num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        SetRuntimeArgs(program,
         void_data_kernel_noc0_read,
         core,
         {dram_buffer2->address(),
         dram_buffer1->address(),
         n_tiles,
         num_tiles_written,
         num_output_tiles_per_core, 
         i});

        SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
        SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer->address(), n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
        num_tiles_written += num_output_tiles_per_core;
    }

    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);
    
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
    //To Enable Profiling
    /////tt_metal::DumpDeviceProfileResults(device, program);

    //Output data
    std::vector<bfloat16> sparse_vec;
    EnqueueReadBuffer(cq, dram_buffer, sparse_vec, true);

#ifdef PRINT_DEBUG  
    printf("Dense_size = %zu pattern_size = %zu sparse_size = %zu\n",dense_vec.size(), pattern_mat_val.size(), sparse_vec.size());
    
    printf("TT Input : \n");
    for(uint32_t i= 0; i < dense.size(); i = i+stride_len){
      printf("%f ", dense_vec[i].to_float());
    }

    printf("\n");
    printf("TT Result : \n");
    //if(n_tiles_rem > 0){
    //  for(uint32_t i= (dense_vec.size() * 2) - 512; i < dense_vec.size() * 2 - 20; i = i+stride_len){
    //    printf("%f ", c_bf16[i].to_float());
    //  }
    // }
    //else {
      for(uint32_t i= sparse_vec.size() - (pattern_length * stride_len); i < sparse_vec.size(); i = i+stride_len){
        printf("%f ", sparse_vec[i].to_float());
      }  
    //}
    printf("\n\n");
#endif
    CloseDevice(device);

} else { //Host Run
  if (timed)
    timer.start();
  printf("%zu %zu %zu %zu\n", count , pattern_length, dense.size(), sparse.size());
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < pattern_length; ++j) {
      sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];
      //if(i > (count - 3))
      //  printf("%f ", sparse[pattern[j] + delta * i]);
    }
  }
  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

}

void Configuration<Spatter::Serial>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::Serial>::multi_gather(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_gather.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j)
    {
      dense[j + pattern_length * (i % wrap)] =
          sparse[pattern[pattern_gather[j]] + delta * i];
    }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::Serial>::multi_scatter(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j){
      sparse[pattern[pattern_scatter[j]] + delta * i] =
          dense[j + pattern_length * (i % wrap)];
    }
  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

#ifdef USE_OPENMP
Configuration<Spatter::OpenMP>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread,
    double *&dev_dense, size_t &dense_size,const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const long int seed,
    const size_t wrap, const size_t count, const int nthreads,
    const unsigned long nruns, const bool aggregate, const bool atomic,
    const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather, delta_scatter, seed, wrap,
          count, 0, 1024, nthreads, nruns, aggregate, atomic, verbosity) {
  ConfigurationBase::setup();
}

int Configuration<Spatter::OpenMP>::run(bool timed, unsigned long run_id) {
  omp_set_num_threads(omp_threads);
  return ConfigurationBase::run(timed, run_id);
}

void Configuration<Spatter::OpenMP>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *sl = sparse.data() + delta * i;
      double *tl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[j] = sl[pattern[j]];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }

}

void Configuration<Spatter::OpenMP>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *tl = sparse.data() + delta * i;
      double *sl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[pattern[j]] = sl[j];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::OpenMP>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    double *tl = sparse_scatter.data() + delta_scatter * i;
    double *sl = sparse_gather.data() + delta_gather * i;

#pragma omp simd
    for (size_t j = 0; j < pattern_length; ++j) {
      tl[pattern_scatter[j]] = sl[pattern_gather[j]];
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::OpenMP>::multi_gather(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_gather.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *sl = sparse.data() + delta * i;
      double *tl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[j] = sl[pattern[pattern_gather[j]]];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}


void Configuration<Spatter::OpenMP>::multi_scatter(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *tl = sparse.data() + delta * i;
      double *sl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[pattern[pattern_scatter[j]]] = sl[j];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}
#endif

#ifdef USE_CUDA
Configuration<Spatter::CUDA>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread, double *&dev_dense,
    size_t &dense_size, const size_t delta, const size_t delta_gather,
    const size_t delta_scatter, const long int seed, const size_t wrap,
    const size_t count, const size_t shared_mem, const size_t local_work_size,
    const unsigned long nruns, const bool aggregate, const bool atomic,
    const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather, delta_scatter, seed,
          wrap, count, shared_mem, local_work_size, 1, nruns, aggregate, atomic,
          verbosity) {
  
  setup();
}

Configuration<Spatter::CUDA>::~Configuration() {
  checkCudaErrors(cudaFree(dev_pattern));
  checkCudaErrors(cudaFree(dev_pattern_gather));
  checkCudaErrors(cudaFree(dev_pattern_scatter));

  if (dev_sparse) {
    checkCudaErrors(cudaFree(dev_sparse));
    dev_sparse = nullptr;
  }

  if (dev_sparse_gather) {
    checkCudaErrors(cudaFree(dev_sparse_gather));
    dev_sparse_gather = nullptr;
  }

  if (dev_sparse_scatter) {
    checkCudaErrors(cudaFree(dev_sparse_scatter));
    dev_sparse_scatter = nullptr;
  }

  if (dev_dense) {
    checkCudaErrors(cudaFree(dev_dense));
    dev_dense = nullptr;
  }
}

int Configuration<Spatter::CUDA>::run(bool timed, unsigned long run_id) {
  return ConfigurationBase::run(timed, run_id);
}

void Configuration<Spatter::CUDA>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_gather_wrapper(
      dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms = cuda_scatter_atomic_wrapper(
        dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);
  else
    time_ms = cuda_scatter_wrapper(
        dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms = cuda_scatter_gather_atomic_wrapper(dev_pattern_scatter,
        dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
        pattern_length, delta_scatter, delta_gather, wrap, count);
  else
    time_ms = cuda_scatter_gather_wrapper(dev_pattern_scatter,
        dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
        pattern_length, delta_scatter, delta_gather, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::multi_gather(
    bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_gather.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_multi_gather_wrapper(dev_pattern, dev_pattern_gather,
      dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::multi_scatter(
    bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms =
        cuda_multi_scatter_atomic_wrapper(dev_pattern, dev_pattern_scatter,
            dev_sparse, dev_dense, pattern_length, delta, wrap, count);
  else
    time_ms = cuda_multi_scatter_wrapper(dev_pattern, dev_pattern_scatter,
        dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::setup() {
  ConfigurationBase::setup();

  checkCudaErrors(
      cudaMalloc((void **)&dev_pattern, sizeof(size_t) * pattern.size()));
  checkCudaErrors(cudaMalloc(
      (void **)&dev_pattern_gather, sizeof(size_t) * pattern_gather.size()));
  checkCudaErrors(cudaMalloc(
      (void **)&dev_pattern_scatter, sizeof(size_t) * pattern_scatter.size()));

  checkCudaErrors(cudaMemcpy(dev_pattern, pattern.data(),
      sizeof(size_t) * pattern.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_pattern_gather, pattern_gather.data(),
      sizeof(size_t) * pattern_gather.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_pattern_scatter, pattern_scatter.data(),
      sizeof(size_t) * pattern_scatter.size(), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaDeviceSynchronize());
}
#endif

} // namespace Spatter