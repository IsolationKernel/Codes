#include <mass_ml/ds/ds_file_csv.h>
#include <mass_ml/ds/ds_mdl_dependent.h>
#include <mass_ml/ds/ds_memory.h>
#include <mass_ml/ds/ds_tmp_file.h>
#include <mass_ml/fs/alg_fs_aNNE.h>
#include <mass_ml/fs/alg_fs_iNNE.h>
#include <mass_ml/math/math.h>
#include <mass_ml/tr/trace.h>
#include <mass_ml/util/parse_parameter.h>

#include <mass_ml/config_fs.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace {

  enum class alg_e : uint8_t {
    aNNE, iNNE, none
  };

  enum class command_e : uint8_t {
    generate, verify, none
  };

  alg_e alg{alg_e::none};
  command_e command{command_e::none};

  std::string target_dir_name;

  std::size_t random_seed;

  void generate_test_set_10D() {
    std::mt19937 mt_random{random_seed};
    std::uniform_real_distribution uniform_dist;

    std::ofstream output;

    std::stringstream is;
    is << target_dir_name << fs::path::preferred_separator << "td_xNNE_10D_10000.csv";

    output.open(is.str());
    output << std::setprecision(std::numeric_limits<double>::max_digits10 + 2);

    for (std::size_t i = 0; i < 10000; i++) {
      for (std::size_t j = 0; j < 9; j++) {
        output << uniform_dist(mt_random) << ",";
      }

      output << uniform_dist(mt_random) << ",C" << std::endl;
    }

    output.close();

    std::shared_ptr<mass_ml::data_source_c> src_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, "td_xNNE_10D_10000.csv");

    std::vector<std::size_t> psi_val{2, 8};
    std::size_t sets = 100;

    for (std::size_t psi : psi_val) {
      std::stringstream is;
      is << "mdl_xNNE_10D_10000_psi_" << psi << "_t_100.csv";
      std::shared_ptr<mass_ml::data_source_c> dst_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, is.str(), mass_ml::ds_file_mode_e::write);
      std::unique_ptr<mass_ml::data_source_c> model_data = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_dependent_c>(random_seed, sets, psi, src_data);

      std::vector<double> row;
      row.resize(10, 0.0);

      for (std::size_t i = 0; i < (psi * sets); i++) {
        std::fill(row.begin(), row.end(), 0.0);
        model_data->load_row(row, i);
        dst_data->store_row(row, "C");
      }
    }
  }

  void parse_command_line(int argc, char ** argv) {
    std::string parameter;

    mass_ml::parse_parameter(argc, argv, "--tmp_dir", parameter, "No temp directory given.");
    mass_ml::init_tmp_dir(parameter);

    mass_ml::parse_parameter(argc, argv, "--target_dir", target_dir_name, "No target directory name given.");

    mass_ml::parse_parameter(argc, argv, "--alg", parameter, "No algorithm given.");
    std::transform(parameter.begin(), parameter.end(), parameter.begin(), [](unsigned char c){return std::tolower(c);});

    if (parameter == "anne") {
      alg = alg_e::aNNE;
    } else if (parameter == "inne") {
      alg = alg_e::iNNE;
    } else {
      ml_tr_initiate("Unknown algorithm - " + parameter + ".");
    }

    mass_ml::parse_parameter(argc, argv, "--cmd", parameter, "No command given.");
    std::transform(parameter.begin(), parameter.end(), parameter.begin(), [](unsigned char c){return std::tolower(c);});

    if (parameter == "generate") {
      command = command_e::generate;
    } else if (parameter == "verify") {
      command = command_e::verify;
    } else {
      ml_tr_initiate("Unknown command - " + parameter + ".");
    }

    mass_ml::parse_parameter(argc, argv, "--random_seed", random_seed, "No random seed given.");

    mass_ml::init_math(4ull * 1024 * 1024 * 1024, 4ull * 1024 * 1024 * 1024);
  }

  void verify_aNNE_10D_psi_2() {
    std::size_t sample_size = 2;
    std::size_t sets = 100;

    std::shared_ptr<mass_ml::data_source_c> src_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, "td_xNNE_10D_10000.csv");
    std::unique_ptr<mass_ml::data_source_c> model_data = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_dependent_c>(random_seed, sets, sample_size, src_data);
    std::unique_ptr<mass_ml::data_source_c> dst_data = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(sample_size * sets);
//    std::unique_ptr<mass_ml::data_source_c> dst_data = mass_ml::data_source_c::make_unique<mass_ml::ds_memory_c>(sample_size * sets);

    std::unique_ptr<mass_ml::alg_feature_space_c> alg = std::make_unique<mass_ml::alg_fs_aNNE_c>(sets, sample_size);
    alg->transform(*dst_data, *src_data, *model_data);

    std::shared_ptr<mass_ml::data_source_c> res_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, "res_aNNE_10D_10000_psi_2_t_100.csv");

    if (dst_data->cols() != res_data->cols()) {
      std::stringstream is;
      is << "Mismatch in number of columns. Should be " << res_data->cols() << " but got " << dst_data->cols();
      ml_tr_initiate(is.str());
    }

    if (dst_data->rows() != res_data->rows()) {
      std::stringstream is;
      is << "Mismatch in number of rows. Should be " << res_data->rows() << " but got " << dst_data->rows();
      ml_tr_initiate(is.str());
    }

    std::vector<double> x;
    x.resize(res_data->cols(), 0.0);
    std::vector<double> y;
    y.resize(dst_data->cols());

    for (std::size_t i = 0; i < dst_data->rows(); i++) {
      std::fill(x.begin(), x.end(), 0.0);
      res_data->load_row(x, i);
      std::fill(y.begin(), y.end(), 0.0);
      dst_data->load_row(y, i);

      for (std::size_t j = 0; j < res_data->cols(); j++) {
        if (x[j] != y[j]) {
          ml_tr_initiate("aNNE failure.");
        }
      }
    }

    std::cout << "aNNE_10D psi = 2 passed!" << std::endl;
  }

  void verify_aNNE_10D_psi_8() {
    std::size_t sample_size = 8;
    std::size_t sets = 100;

    std::shared_ptr<mass_ml::data_source_c> src_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, "td_xNNE_10D_10000.csv");
    std::unique_ptr<mass_ml::data_source_c> model_data = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_dependent_c>(random_seed, sets, sample_size, src_data);
    std::unique_ptr<mass_ml::data_source_c> dst_data = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(sample_size * sets);

    std::unique_ptr<mass_ml::alg_feature_space_c> alg = std::make_unique<mass_ml::alg_fs_aNNE_c>(sets, sample_size);
    alg->transform(*dst_data, *src_data, *model_data);

    std::shared_ptr<mass_ml::data_source_c> res_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, "res_aNNE_10D_10000_psi_8_t_100.csv");

    if (dst_data->cols() != res_data->cols()) {
      std::stringstream is;
      is << "Mismatch in number of columns. Should be " << res_data->cols() << " but got " << dst_data->cols();
      ml_tr_initiate(is.str());
    }

    if (dst_data->rows() != res_data->rows()) {
      std::stringstream is;
      is << "Mismatch in number of rows. Should be " << res_data->rows() << " but got " << dst_data->rows();
      ml_tr_initiate(is.str());
    }

    std::vector<double> x;
    x.resize(res_data->cols(), 0.0);
    std::vector<double> y;
    y.resize(dst_data->cols());

    for (std::size_t i = 0; i < dst_data->rows(); i++) {
      std::fill(x.begin(), x.end(), 0.0);
      res_data->load_row(x, i);
      std::fill(y.begin(), y.end(), 0.0);
      dst_data->load_row(y, i);

      for (std::size_t j = 0; j < res_data->cols(); j++) {
        if (x[j] != y[j]) {
          std::cerr << "i = " << i << ", j = " << j << " : " << x[j] << " - " << y[j] << std::endl;
          ml_tr_initiate("aNNE failure.");
        }
      }
    }

    std::cout << "aNNE_10D psi = 8 passed!" << std::endl;
  }

  void verify_iNNE_10D_psi_2() {
    std::size_t sample_size = 2;
    std::size_t sets = 100;

    std::shared_ptr<mass_ml::data_source_c> src_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, "td_xNNE_10D_10000.csv");
    std::unique_ptr<mass_ml::data_source_c> model_data = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_dependent_c>(random_seed, sets, sample_size, src_data);
    std::unique_ptr<mass_ml::data_source_c> dst_data = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(sample_size * sets);

    std::unique_ptr<mass_ml::alg_feature_space_c> alg = std::make_unique<mass_ml::alg_fs_iNNE_c>(sets, sample_size);
    alg->transform(*dst_data, *src_data, *model_data);

    std::shared_ptr<mass_ml::data_source_c> res_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, "res_iNNE_10D_10000_psi_2_t_100.csv");

    if (dst_data->cols() != res_data->cols()) {
      std::stringstream is;
      is << "Mismatch in number of columns. Should be " << res_data->cols() << " but got " << dst_data->cols();
      ml_tr_initiate(is.str());
    }

    if (dst_data->rows() != res_data->rows()) {
      std::stringstream is;
      is << "Mismatch in number of rows. Should be " << res_data->rows() << " but got " << dst_data->rows();
      ml_tr_initiate(is.str());
    }

    std::vector<double> x;
    x.resize(res_data->cols(), 0.0);
    std::vector<double> y;
    y.resize(dst_data->cols());

    for (std::size_t i = 0; i < dst_data->rows(); i++) {
      std::fill(x.begin(), x.end(), 0.0);
      res_data->load_row(x, i);
      std::fill(y.begin(), y.end(), 0.0);
      dst_data->load_row(y, i);

      for (std::size_t j = 0; j < res_data->cols(); j++) {
        if (x[j] != y[j]) {
          ml_tr_initiate("aNNE failure.");
        }
      }
    }

    std::cout << "iNNE_10D psi = 2 passed!" << std::endl;
  }

  void verify_iNNE_10D_psi_8() {
    std::size_t sample_size = 8;
    std::size_t sets = 100;

    std::shared_ptr<mass_ml::data_source_c> src_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, "td_xNNE_10D_10000.csv");
    std::unique_ptr<mass_ml::data_source_c> model_data = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_dependent_c>(random_seed, sets, sample_size, src_data);
    std::unique_ptr<mass_ml::data_source_c> dst_data = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(sample_size * sets);

    std::unique_ptr<mass_ml::alg_feature_space_c> alg = std::make_unique<mass_ml::alg_fs_iNNE_c>(sets, sample_size);
    alg->transform(*dst_data, *src_data, *model_data);

    std::shared_ptr<mass_ml::data_source_c> res_data = mass_ml::data_source_c::make_shared<mass_ml::ds_file_csv_c>(target_dir_name, "res_iNNE_10D_10000_psi_8_t_100.csv");

    if (dst_data->cols() != res_data->cols()) {
      std::stringstream is;
      is << "Mismatch in number of columns. Should be " << res_data->cols() << " but got " << dst_data->cols();
      ml_tr_initiate(is.str());
    }

    if (dst_data->rows() != res_data->rows()) {
      std::stringstream is;
      is << "Mismatch in number of rows. Should be " << res_data->rows() << " but got " << dst_data->rows();
      ml_tr_initiate(is.str());
    }

    std::vector<double> x;
    x.resize(res_data->cols(), 0.0);
    std::vector<double> y;
    y.resize(dst_data->cols());

    for (std::size_t i = 0; i < dst_data->rows(); i++) {
      std::fill(x.begin(), x.end(), 0.0);
      res_data->load_row(x, i);
      std::fill(y.begin(), y.end(), 0.0);
      dst_data->load_row(y, i);

      for (std::size_t j = 0; j < res_data->cols(); j++) {
        if (x[j] != y[j]) {
          std::cerr << "i = " << i << ", j = " << j << " : " << x[j] << " - " << y[j] << std::endl;
          ml_tr_initiate("aNNE failure.");
        }
      }
    }

    std::cout << "iNNE_10D psi = 8 passed!" << std::endl;
  }

}

int main(int argc, char ** argv) {
  try {
    parse_command_line(argc, argv);

    if (command == command_e::generate) {
      generate_test_set_10D();
    } else {
      if (alg == alg_e::aNNE) {
        verify_aNNE_10D_psi_2();
        verify_aNNE_10D_psi_8();
      } else {
        verify_iNNE_10D_psi_2();
        verify_iNNE_10D_psi_8();
      }
    }
  } catch (std::exception const & ex) {
    ml_tr_handle(ex);
    std::cerr << trace::latest() << std::endl;
  }

  return EXIT_SUCCESS;
}
