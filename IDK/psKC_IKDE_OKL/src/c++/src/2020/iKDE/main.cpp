#include <mass_ml/ds/ds_file_csv.h>
#include <mass_ml/ds/ds_file_libsvm.h>
#include <mass_ml/ds/ds_mdl_dependent.h>
#include <mass_ml/ds/ds_mdl_independent.h>
#include <mass_ml/ds/ds_tmp_file.h>
#include <mass_ml/eval/auc.h>
#include <mass_ml/fs/alg_fs_aNNE.h>
#include <mass_ml/fs/alg_fs_iNNE.h>
#include <mass_ml/math/math.h>
#include <mass_ml/tr/trace.h>
#include <mass_ml/util/parse_parameter.h>

#include <iostream>
#include <fstream>
#include <sstream>

namespace {

  enum class alg_e : uint8_t {
    aNNE, iNNE, DotProduct, none
  };

  enum class data_file_format_e : uint8_t {
    CSV, LibSVM, none
  };

  enum class data_model_e : uint8_t {
    dependent, independent, none
  };

  alg_e alg{alg_e::none};
  data_file_format_e dff{data_file_format_e::none};
  data_model_e data_model{data_model_e::none};

  std::size_t random_seed;
  std::size_t sample_size;
  std::size_t sets;

  std::string src_dir_name;
  std::string src_file_name;

  std::shared_ptr<mass_ml::data_source_c> data{nullptr};

  std::vector<double> scores; // scores are ranked in descending order

  void compute_D(std::vector<double> & psi_D) {
    psi_D.clear();
    psi_D.resize(sets * sample_size, 0.0);

    std::vector<double> row;
    row.resize(sample_size * sets, 0.0);

    for (std::size_t x = 0; x < data->rows(); x++) {
      std::fill(row.begin(), row.end(), 0.0);
      data->load_row(row, x);

      for (std::size_t i = 0; i < row.size(); i++) {
        psi_D[i] += row[i];
      }
    }

    for (std::size_t i = 0; i < psi_D.size(); i++) {
      psi_D[i] /= data->rows();
    }
  }

  void compute_scores(std::vector<double> const & psi_D) {
    std::vector<double> row;
    row.resize(sample_size * sets, 0.0);

    for (std::size_t x = 0; x < data->rows(); x++) {
      std::fill(row.begin(), row.end(), 0.0);
      data->load_row(row, x);

      double sim = 0.0;
 
      for (std::size_t i = 0; i < row.size(); i++) {
        sim += row[i] * psi_D[i];
      }

      sim /= sets;
      scores.push_back(1.0 - sim);
    }
  }

  void go() {
    std::vector<double> psi_D;
    compute_D(psi_D);
    compute_scores(psi_D);
  }

  void load_data() {
    if (alg == alg_e::DotProduct) {
      if (dff == data_file_format_e::CSV) {
        data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(src_dir_name, src_file_name);
      } else {
        data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_libsvm_c>(src_dir_name, src_file_name);
      }
    } else {
      std::cout << "Preprocessing data ..." << std::endl;

      std::shared_ptr<mass_ml::data_source_c> initial_data{nullptr};

      if (dff == data_file_format_e::CSV) {
        initial_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(src_dir_name, src_file_name);
      } else {
        initial_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_libsvm_c>(src_dir_name, src_file_name);
      }

      std::unique_ptr<mass_ml::data_source_c> model_data;

      if (data_model == data_model_e::dependent) {
        model_data = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_dependent_c>(random_seed, sets, sample_size, initial_data);
      } else {
        model_data = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_independent_c>(random_seed, sets * sample_size, initial_data->cols());
      }

      std::unique_ptr<mass_ml::alg_feature_space_c> alg_;

      if (alg == alg_e::aNNE) {
        alg_ = std::make_unique<mass_ml::alg_fs_aNNE_c>(sets, sample_size);
      } else {
        alg_ = std::make_unique<mass_ml::alg_fs_iNNE_c>(sets, sample_size);
      }

      data = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(sample_size * sets);

      std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

      alg_->transform(*data, *initial_data, *model_data);

      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      double time = elapsed.count();

      std::cout << "Data transform time: " << time << " secs" << std::endl;
    }
  }

  void parse_command_line(int argc, char ** argv) {
    std::string parameter;

    mass_ml::parse_parameter(argc, argv, "--tmp_dir", parameter, "No temp directory given.");
    mass_ml::init_tmp_dir(parameter);

    mass_ml::parse_parameter(argc, argv, "--alg", parameter, "No algorithm given.");
    std::transform(parameter.begin(), parameter.end(), parameter.begin(), [](unsigned char c){return std::tolower(c);});

    if (parameter == "anne") {
      alg = alg_e::aNNE;
    } else if (parameter == "inne") {
      alg = alg_e::iNNE;
    } else if (parameter == "dotproduct") {
      alg = alg_e::DotProduct;
    } else {
      ml_tr_initiate("Unknown algorithm - " + parameter + ".");
    }

    mass_ml::parse_parameter(argc, argv, "--data_file_format", parameter, "No data file format given.");
    std::transform(parameter.begin(), parameter.end(), parameter.begin(), [](unsigned char c){return std::tolower(c);});

    if (parameter == "csv") {
      dff = data_file_format_e::CSV;
    } else if (parameter == "libsvm") {
      dff = data_file_format_e::LibSVM;
    } else {
      ml_tr_initiate("Unknown data file format - " + parameter + ".");
    }

    mass_ml::parse_parameter(argc, argv, "--src_dir", src_dir_name, "No source directory name given.");
    mass_ml::parse_parameter(argc, argv, "--src_file_name", src_file_name, "No source file name given.");

    if ((alg == alg_e::aNNE) || (alg == alg_e::iNNE)) {
      mass_ml::parse_parameter(argc, argv, "--random_seed", random_seed, "No random seed given.");
      mass_ml::parse_parameter(argc, argv, "--sample_size", sample_size, "No sample size given.");
      mass_ml::parse_parameter(argc, argv, "--sets", sets, "No sets given.");
    }

    mass_ml::parse_parameter(argc, argv, "--data_model", parameter, "No data model given.");
    std::transform(parameter.begin(), parameter.end(), parameter.begin(), [](unsigned char c){return std::tolower(c);});

    if (parameter == "dependent") {
      data_model = data_model_e::dependent;
    } else if (parameter == "independent") {
      data_model = data_model_e::independent;
    } else {
      ml_tr_initiate("Unknown data mode - " + parameter + ".");
    }

    mass_ml::init_math(4ull * 1024 * 1024 * 1024, 4ull * 1024 * 1024 * 1024);
  }

}

int main(int argc, char ** argv) {
  try {
    parse_command_line(argc, argv);
    load_data();

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    go();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    std::cout << "Alg time: " << time << " secs" << std::endl;
    std::cout << std::endl << std::endl;

	

    mass_ml::auc(std::cout, *data, scores, src_file_name, sample_size);
  } catch (std::exception const & ex) {
    ml_tr_handle(ex);
    std::cerr << trace::latest() << std::endl;
  }

  mass_ml::clean_up_tmp_dir();

  return EXIT_SUCCESS;
}
