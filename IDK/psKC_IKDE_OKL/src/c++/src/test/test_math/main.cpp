#include <mass_ml/ds/ds_file_csv.h>
#include <mass_ml/ds/ds_memory.h>
#include <mass_ml/ds/ds_tmp_file.h>
#include <mass_ml/math/math.h>
#include <mass_ml/tr/trace.h>
#include <mass_ml/util/parse_parameter.h>

#include <mass_ml/config_fs.h>

#include <iomanip>
#include <iostream>
#include <random>

#include <cassert>
#include <cmath>

/*
 * Code to check "if two matrix are the same" are taken from the following web-site:
 *
 * https://stackoverflow.com/questions/50781723/how-to-check-if-two-matrices-are-the-same
 *
 */

namespace {

  enum class command_e : uint8_t {
    generate, verify, none
  };

  command_e command{command_e::none};

  std::string target_dir_name;

  std::size_t random_seed;

  void generate_test_set_1D() {
    std::mt19937 mt_random{random_seed};
    std::uniform_real_distribution uniform_dist;

    std::vector<std::string> sets{"a", "b"};

    for (std::string id : sets) {
      std::ofstream output;

      std::stringstream is;
      is << target_dir_name << fs::path::preferred_separator << "td_math_1D_10" << id << ".csv";

      output.open(is.str());
      output << std::setprecision(std::numeric_limits<double>::max_digits10 + 2);

      for (std::size_t i = 0; i < 10; i++) {
        output << uniform_dist(mt_random) << std::endl;
      }
    }
  }

  void generate_test_set_2D() {
    std::mt19937 mt_random{random_seed};
    std::uniform_real_distribution uniform_dist;

    std::vector<std::string> sets{"a", "b"};

    for (std::string id : sets) {
      std::ofstream output;

      std::stringstream is;
      is << target_dir_name << fs::path::preferred_separator << "td_math_2D_100" << id << ".csv";

      output.open(is.str());
      output << std::setprecision(std::numeric_limits<double>::max_digits10 + 2);

      for (std::size_t i = 0; i < 100; i++) {
        output << uniform_dist(mt_random) << "," << uniform_dist(mt_random) << std::endl;
      }
    }
  }

  void generate_test_set_10D() {
    std::mt19937 mt_random{random_seed};
    std::uniform_real_distribution uniform_dist;

    std::vector<std::string> sets{"a", "b"};

    for (std::string id : sets) {
      std::ofstream output;

      std::stringstream is;
      is << target_dir_name << fs::path::preferred_separator << "td_math_10D_1000" << id << ".csv";

      output.open(is.str());
      output << std::setprecision(std::numeric_limits<double>::max_digits10 + 2);

      for (std::size_t i = 0; i < 1000; i++) {
        for (std::size_t j = 0; j < 9; j++) {
          output << uniform_dist(mt_random) << ",";
        }

        output << uniform_dist(mt_random) << std::endl;
      }
    }
  }

  void generate_test_set_1000D() {
    std::mt19937 mt_random{random_seed};
    std::uniform_real_distribution uniform_dist;

    std::vector<std::string> sets{"a", "b"};
    std::vector<std::size_t> sizes{10000, 20000};

    for (std::size_t i = 0; i < sets.size(); i++) {
      std::ofstream output;

      std::stringstream is;
      is << target_dir_name << fs::path::preferred_separator << "td_math_1000D_" << sizes[i] << sets[i] << ".csv";

      output.open(is.str());
      output << std::setprecision(std::numeric_limits<double>::max_digits10 + 2);

      for (std::size_t j = 0; j < sizes[i]; j++) {
        for (std::size_t k = 0; k < 999; k++) {
          output << uniform_dist(mt_random) << ",";
        }

        output << uniform_dist(mt_random) << std::endl;
      }
    }
  }

  void parse_command_line(int argc, char ** argv) {
    std::string parameter;

    mass_ml::parse_parameter(argc, argv, "--target_dir", target_dir_name, "No target directory name given.");

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
//    mass_ml::init_math(4ull * 1024 * 1024 * 1024,  1024 * 1024 * 1024 / 64); // use this to check gpu blocking code
  }

  inline double reldiff(double a, double b) {
    double divisor = std::fmax(std::fabs(a), std::fabs(b)); /* If divisor is zero, both x and y are zero, so the difference between them is zero */

    if (divisor == 0.0) {
      return 0.0;
    }

    return std::fabs(a - b) / divisor;
  }

  inline bool double_equals(double a, double b, double rel_diff) {
    return reldiff(a, b) < rel_diff;
  }

  bool verify(mass_ml::data_source_c & reference, mass_ml::data_source_c & data, double epsilon) {
    assert(epsilon >= 0);

    std::vector<double> x;
    x.resize(data.cols(), 0.0);
    std::vector<double> y;
    y.resize(data.cols(), 0.0);

    for (std::size_t i = 0; i < reference.rows(); ++i) {
      std::fill(x.begin(), x.end(), 0.0);
      reference.load_row(x, i);
      std::fill(y.begin(), y.end(), 0.0);
      data.load_row(y, i);

      for (std::size_t j = 0; j < data.cols(); j++) {
        if (!double_equals(x[i], y[i], epsilon)) {
          std::cout << "Mismatch - Row: " << i << ", col: " << j << " - " << x[j] << " " << y[j] << std::endl;
          return false;
        }
      }
    }

    return true;
  }

  void verify_test_NN_1D() {
    std::unique_ptr<mass_ml::data_source_c> td_a = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_1D_10a.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_b = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_1D_10b.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_test = mass_ml::data_source_c::make_unique<mass_ml::ds_memory_c>(10);

    mass_ml::compute_distance(*td_test, *td_a, *td_b);

    std::vector<std::size_t> psi_val{10, 5};
    std::vector<std::size_t> sets_vals{1, 2};

    for (std::size_t i = 0; i < psi_val.size(); i++) {
      std::size_t psi = psi_val[i];
      std::size_t sets = sets_vals[i];

      std::stringstream is;
      is << "res_math_1D_nn_psi_" << psi << ".csv";
      std::unique_ptr<mass_ml::data_source_c> td_res = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, is.str(), mass_ml::ds_file_mode_e::read, true, true);

      std::unique_ptr<mass_ml::data_source_c> idx = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(sets);
      mass_ml::find_nearest_neighbour(*td_test, *idx, psi);

      std::vector<double> x;
      x.resize(td_res->cols(), 0.0);
      std::vector<double> y;
      y.resize(idx->cols(), 0.0);

      for (std::size_t i = 0; i < td_res->rows(); i++) {
        std::fill(x.begin(), x.end(), 0.0);
        td_res->load_row(x, i);
        std::fill(y.begin(), y.end(), 0.0);
        idx->load_row(y, i);

        for (std::size_t j = 0; j < sets; j++) {
          if (x[j] != y[j]) {
            ml_tr_initiate("Nearest Neighbour mismatch!");
          }
        }
      }

      std::cout << "test_NN_1D (psi = " << psi << ") passed!" << std::endl;
    }
  }

  void verify_test_set_1D() {
    std::unique_ptr<mass_ml::data_source_c> td_a = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_1D_10a.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_b = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_1D_10b.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_res = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "res_math_1D_dist_p_2.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_test = mass_ml::data_source_c::make_unique<mass_ml::ds_memory_c>(10);

    mass_ml::compute_distance(*td_test, *td_a, *td_b);

    if (td_test->cols() != td_res->cols()) {
      std::stringstream is;
      is << "Mismatch in number of columns. Should be " << td_res->cols() << " but got " << td_test->cols();
      ml_tr_initiate(is.str());
    }

    if (td_test->rows() != td_res->rows()) {
      std::stringstream is;
      is << "Mismatch in number of rows. Should be " << td_res->rows() << " but got " << td_test->rows();
      ml_tr_initiate(is.str());
    }

    if (!verify(*td_res, *td_test, 1.0e-6)) {
      ml_tr_initiate("Distance calculation failure.");
    }

    std::cout << "test_set_1D passed!" << std::endl;
  }

  void verify_test_set_2D() {
    std::unique_ptr<mass_ml::data_source_c> td_a = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_2D_100a.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_b = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_2D_100b.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_res = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "res_math_2D_dist_p_2.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_test = mass_ml::data_source_c::make_unique<mass_ml::ds_memory_c>(100);

    mass_ml::compute_distance(*td_test, *td_a, *td_b);

    if (td_test->cols() != td_res->cols()) {
      std::stringstream is;
      is << "Mismatch in number of columns. Should be " << td_res->cols() << " but got " << td_test->cols();
      ml_tr_initiate(is.str());
    }

    if (td_test->rows() != td_res->rows()) {
      std::stringstream is;
      is << "Mismatch in number of rows. Should be " << td_res->rows() << " but got " << td_test->rows();
      ml_tr_initiate(is.str());
    }

    if (!verify(*td_res, *td_test, 1.0e-6)) {
      ml_tr_initiate("Distance calculation failure.");
    }

    std::cout << "test_set_2D passed!" << std::endl;
  }

  void verify_test_set_10D() {
    std::unique_ptr<mass_ml::data_source_c> td_a = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_10D_1000a.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_b = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_10D_1000b.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_res = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "res_math_10D_dist_p_2.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_test = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(1000);

    mass_ml::compute_distance(*td_test, *td_a, *td_b);

    if (td_test->cols() != td_res->cols()) {
      std::stringstream is;
      is << "Mismatch in number of columns. Should be " << td_res->cols() << " but got " << td_test->cols();
      ml_tr_initiate(is.str());
    }

    if (td_test->rows() != td_res->rows()) {
      std::stringstream is;
      is << "Mismatch in number of rows. Should be " << td_res->rows() << " but got " << td_test->rows();
      ml_tr_initiate(is.str());
    }

    if (!verify(*td_res, *td_test, 1.0e-6)) {
      ml_tr_initiate("Distance calculation failure.");
    }

    std::cout << "test_set_10D passed!" << std::endl;
  }

  void verify_test_set_1000D() {
    std::unique_ptr<mass_ml::data_source_c> td_a = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_1000D_10000a.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_b = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "td_math_1000D_20000b.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_res = mass_ml::data_source_c::make_unique<mass_ml::ds_file_csv_c>(target_dir_name, "res_math_1000D_dist_p_2.csv", mass_ml::ds_file_mode_e::read, true, true);
    std::unique_ptr<mass_ml::data_source_c> td_test = mass_ml::data_source_c::make_unique<mass_ml::ds_tmp_file_c>(20000);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    mass_ml::compute_distance(*td_test, *td_a, *td_b);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    std::cout << "Alg time: " << time << " secs" << std::endl;
    std::cout << std::endl;

    if (td_test->cols() != td_res->cols()) {
      std::stringstream is;
      is << "Mismatch in number of columns. Should be " << td_res->cols() << " but got " << td_test->cols();
      ml_tr_initiate(is.str());
    }

    if (td_test->rows() != td_res->rows()) {
      std::stringstream is;
      is << "Mismatch in number of rows. Should be " << td_res->rows() << " but got " << td_test->rows();
      ml_tr_initiate(is.str());
    }

    if (!verify(*td_res, *td_test, 1.0e-6)) {
      ml_tr_initiate("Distance calculation failure.");
    }

    std::cout << "test_set_1000D passed!" << std::endl;
  }

}

int main(int argc, char ** argv) {
  try {
    parse_command_line(argc, argv);

    if (command == command_e::generate) {
      generate_test_set_1D();
      generate_test_set_2D();
      generate_test_set_10D();
      generate_test_set_1000D();
    } else {
      verify_test_set_1D();
      verify_test_set_2D();
      verify_test_set_10D();
      verify_test_set_1000D();

      verify_test_NN_1D();
    }
  } catch (std::exception const & ex) {
    ml_tr_handle(ex);
    std::cerr << trace::latest() << std::endl;
  }

  return EXIT_SUCCESS;
}
