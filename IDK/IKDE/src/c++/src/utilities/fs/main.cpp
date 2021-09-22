#include <mass_ml/ds/ds_file_libsvm.h>
#include <mass_ml/ds/ds_mdl_dependent.h>
#include <mass_ml/ds/ds_memory.h>
#include <mass_ml/fs/alg_fs_aNNE.h>
#include <mass_ml/fs/alg_fs_iNNE.h>
#include <mass_ml/math/math.h>
#include <mass_ml/tr/trace.h>
#include <mass_ml/util/parse_parameter.h>

#include <iostream>

namespace {

  enum class alg_e : uint8_t {
    aNNE, iNNE, none
  };

  enum class data_set_e : uint8_t {
    single, train_test, none
  };

  alg_e alg{alg_e::none};
  data_set_e data_set{data_set_e::none};

  std::unique_ptr<mass_ml::data_source_c> dst_training_data{nullptr};
  std::shared_ptr<mass_ml::data_source_c> src_training_data{nullptr};

  std::unique_ptr<mass_ml::data_source_c> dst_testing_data{nullptr};
  std::unique_ptr<mass_ml::data_source_c> src_testing_data{nullptr};

  std::unique_ptr<mass_ml::data_source_c> model_data{nullptr};

  std::size_t random_seed;
  std::size_t sample_size;
  std::size_t sets;

  std::unique_ptr<mass_ml::data_source_c> preload_file(std::string const & dir_name, std::string const & file_name) {
    std::unique_ptr<mass_ml::data_source_c> data = mass_ml::data_source_c::make_unique<mass_ml::ds_file_libsvm_c>(dir_name, file_name);
    std::unique_ptr<mass_ml::data_source_c> mem = mass_ml::data_source_c::make_unique<mass_ml::ds_memory_c>(data->cols());

    std::vector<double> row;
    row.resize(data->cols(), 0.0);

    for (std::size_t i = 0; i < data->rows(); i++) {
      std::fill(row.begin(), row.end(), 0.0);
      data->load_row(row, i);
      mem->store_row(row, data->label(i));
    }

    return mem;
  }

  void parse_command_line(int argc, char ** argv) {
    std::string parameter;

    mass_ml::parse_parameter(argc, argv, "--alg", parameter, "No algorithm given.");
    std::transform(parameter.begin(), parameter.end(), parameter.begin(), [](unsigned char c){return std::tolower(c);});

    if (parameter == "anne") {
      alg = alg_e::aNNE;
    } else if (parameter == "inne") {
      alg = alg_e::iNNE;
    } else {
      ml_tr_initiate("Unknown algorithm - " + parameter + ".");
    }

    mass_ml::parse_parameter(argc, argv, "--data_set", parameter, "No data set given.");
    std::transform(parameter.begin(), parameter.end(), parameter.begin(), [](unsigned char c){return std::tolower(c);});

    if (parameter == "single") {
      data_set = data_set_e::single;
    } else if (parameter == "train_test") {
      data_set = data_set_e::train_test;
    } else {
      ml_tr_initiate("Unknown data set - " + parameter + ".");
    }

    bool preload{false};

    mass_ml::parse_parameter(argc, argv, "--preload", preload);

    std::string dst_dir_name;
    std::string src_dir_name;

    mass_ml::parse_parameter(argc, argv, "--src_dir", src_dir_name, "No source directory name given.");
    mass_ml::parse_parameter(argc, argv, "--dst_dir", dst_dir_name, "No destination directory name given.");

    mass_ml::parse_parameter(argc, argv, "--src_train_file", parameter, "No (src) training file name given.");
    src_training_data = preload ? preload_file(src_dir_name, parameter) : mass_ml::data_source_c::make_shared<mass_ml::ds_file_libsvm_c>(src_dir_name, parameter);

    mass_ml::parse_parameter(argc, argv, "--dst_train_file", parameter, "No (dst) training file name given.");
    dst_training_data = mass_ml::data_source_c::make_unique<mass_ml::ds_file_libsvm_c>(dst_dir_name, parameter, mass_ml::ds_file_mode_e::write);

    if (data_set == data_set_e::train_test) {
      mass_ml::parse_parameter(argc, argv, "--src_test_file", parameter, "No (src) testing file name given.");
      src_testing_data = preload ? preload_file(src_dir_name, parameter) : mass_ml::data_source_c::make_unique<mass_ml::ds_file_libsvm_c>(src_dir_name, parameter);

      mass_ml::parse_parameter(argc, argv, "--dst_test_file", parameter, "No (dst) testing file name given.");
      dst_testing_data = mass_ml::data_source_c::make_unique<mass_ml::ds_file_libsvm_c>(dst_dir_name, parameter, mass_ml::ds_file_mode_e::write);
    }

    mass_ml::parse_parameter(argc, argv, "--random_seed", random_seed, "No random seed given.");
    mass_ml::parse_parameter(argc, argv, "--sample_size", sample_size, "No sample size given.");
    mass_ml::parse_parameter(argc, argv, "--sets", sets, "No sets given.");

    model_data = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_dependent_c>(random_seed, sets, sample_size, src_training_data);

    mass_ml::init_math(4ull * 1024 * 1024 * 1024, 4ull * 1024 * 1024 * 1024);
  }

}

int main(int argc, char ** argv) {
  try {
    parse_command_line(argc, argv);

    std::unique_ptr<mass_ml::alg_feature_space_c> alg_;

    if (alg == alg_e::aNNE) {
      alg_ = std::make_unique<mass_ml::alg_fs_aNNE_c>(sets, sample_size);
    } else {
      alg_ = std::make_unique<mass_ml::alg_fs_iNNE_c>(sets, sample_size);
    }

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    alg_->transform(*dst_training_data, *src_training_data, *model_data);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();

    std::cout << "Training transform time: " << time << " secs" << std::endl;

    if (data_set == data_set_e::train_test) {
      start = std::chrono::steady_clock::now();

      alg_->transform(*dst_testing_data, *src_testing_data, *model_data);

      end = std::chrono::steady_clock::now();
      elapsed = end - start;
      time = elapsed.count();

      std::cout << "Testing transform time: " << time << " secs" << std::endl;

    }
  } catch (std::exception const & ex) {
    ml_tr_handle(ex);
    std::cerr << trace::latest() << std::endl;
  }

  return EXIT_SUCCESS;
}
