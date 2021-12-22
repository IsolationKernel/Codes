#include <mass_ml/ds/ds_file_csv.h>
#include <mass_ml/ds/ds_file_libsvm.h>
#include <mass_ml/ds/ds_mdl_dependent.h>
#include <mass_ml/ds/ds_tmp_file.h>
#include <mass_ml/eval/f1_measure.h>
#include <mass_ml/fs/alg_fs_aNNE.h>
#include <mass_ml/fs/alg_fs_iNNE.h>
#include <mass_ml/math/math.h>
#include <mass_ml/tr/trace.h>
#include <mass_ml/util/parse_parameter.h>

#include <iostream>
#include <set>

namespace {

  enum class alg_e : uint8_t {
    aNNE, iNNE, DotProduct, none
  };

  enum class data_file_format_e : uint8_t {
    CSV, LibSVM, none
  };

  alg_e alg{alg_e::none};
  data_file_format_e dff{data_file_format_e::none};

  std::size_t random_seed;
  std::size_t sample_size;
  std::size_t sets;

  double rho{0.1};
  double tau{0.8};

  std::vector<std::vector<std::size_t>> clusters;
  std::vector<std::size_t> noise;

  std::string src_dir_name;
  std::string src_file_name;

  std::shared_ptr<mass_ml::data_source_c> data{nullptr};

  void compute_psi_D(std::vector<double> & psi_D, std::set<std::size_t> const & D) {
    psi_D.clear();
    psi_D.resize(sets * sample_size, 0.0);

    std::vector<double> row;
    row.resize(sample_size * sets, 0.0);

    for (std::size_t x : D) {
      std::fill(row.begin(), row.end(), 0.0);
      data->load_row(row, x);

      for (std::size_t i = 0; i < row.size(); i++) {
        psi_D[i] += row[i];
      }
    }

    for (std::size_t i = 0; i < psi_D.size(); i++) {
      psi_D[i] /= D.size();
    }
  }

  double compute_score(std::size_t idx, std::vector<double> const & pt) {
    std::vector<double> row;
    row.resize(sample_size * sets, 0.0);
    data->load_row(row, idx);

    double sim = 0.0;

    for (std::size_t i = 0; i < row.size(); i++) {
      sim += row[i] * pt[i];
    }

    return sim / sets;
  }

  void compute_set(std::vector<std::size_t> const & set, std::vector<double> & pt) {
    pt.clear();
    pt.resize(sets * sample_size, 0.0);

    std::vector<double> row;
    row.resize(sample_size * sets, 0.0);

    for (std::size_t idx : set) {
      std::fill(row.begin(), row.end(), 0.0);
      data->load_row(row, idx);

      for (std::size_t i = 0; i < row.size(); i++) {
        pt[i] += row[i];
      }
    }

    for (std::size_t i = 0; i < pt.size(); i++) {
      pt[i] /= set.size();
    }
  }

  std::size_t find_max(std::vector<double> const & psi_D, std::set<std::size_t> const & D) {
    std::size_t x_p = 0;
    double best_sim = 0.0;

    std::vector<double> row;
    row.resize(sample_size * sets, 0.0);

    for (std::size_t x : D) {
      std::fill(row.begin(), row.end(), 0.0);
      data->load_row(row, x);

      double sim = 0.0;

      for (std::size_t i = 0; i < row.size(); i++) {
        sim += psi_D[i] * row[i];
      }

      sim /= sets;

      if (sim > best_sim) {
        best_sim = sim;
        x_p = x;
      }
    }

    return x_p;
  }

  double find_max(std::size_t const x_p, std::size_t & x_q, std::set<std::size_t> const & D) {
    double best_sim = 0.0;
    x_q = 0;

    std::vector<double> row_x_p;
    row_x_p.resize(sample_size * sets, 0.0);
    data->load_row(row_x_p, x_p);

    std::vector<double> row_x;
    row_x.resize(sample_size * sets, 0.0);

    for (std::size_t x : D) {
      std::fill(row_x.begin(), row_x.end(), 0.0);
      data->load_row(row_x, x);

      double sim = 0.0;

      for (std::size_t i = 0; i < row_x.size(); i++) {
        sim += row_x[i] * row_x_p[i];
      }

      if (sim > best_sim) {
        best_sim = sim;
        x_q = x;
      }
    }

    return best_sim / sets;
  }

  void find_min_score(std::size_t & min_idx, double & min_score, std::vector<std::size_t> const & G_k, std::vector<double> const & pt) {
    min_idx = 0;
    min_score = 10.0;

    for (std::size_t x : G_k) {
      double score = compute_score(x, pt);

      if (score < min_score) {
        min_score = score;
        min_idx = x;
      }
    }
  }

  void refine_step() {
    std::vector<std::vector<double>> G_pt;

    for (std::vector<std::size_t> & G_k : clusters) {
      std::vector<double> pt(sets * sample_size, 0.0);
      compute_set(G_k, pt);
      G_pt.push_back(pt);
    }

    for (std::size_t i = 0; i < clusters.size(); i++) {
      std::vector<std::size_t> & G_k_i = clusters[i];
      std::vector<double> & pt_i = G_pt[i];

      bool finished = false;

      while (!finished) {
        finished = true;
        std::size_t min_idx;
        double min_score;

        find_min_score(min_idx, min_score, G_k_i, pt_i);

        std::size_t best_idx = 0;

        for (std::size_t j = 0; j < clusters.size(); j++) {
          if (i != j) {
            double score = compute_score(min_idx, G_pt[j]);

            if (score > min_score) {
              min_score = score;
              best_idx = j;
              finished = false;
            }
          }
        }

        if (!finished) {
          std::vector<std::size_t> & G_k_j = clusters[best_idx];
          std::vector<double> & pt_j = G_pt[best_idx];

          G_k_i.erase(std::remove(G_k_i.begin(), G_k_i.end(), min_idx), G_k_i.end());
          G_k_j.push_back(min_idx);

          compute_set(G_k_i, pt_i);
          compute_set(G_k_j, pt_j);
        }
      }
    }
  }

  void go() {
    std::vector<std::size_t> D_temp(data->rows());
    std::iota(D_temp.begin(), D_temp.end(), 0);
    std::set<std::size_t> D(D_temp.begin(), D_temp.end());

    std::vector<double> psi_D;

    while (D.size() > 1) {
      compute_psi_D(psi_D, D);

      std::size_t x_p = find_max(psi_D, D);
      D.erase(x_p);
      std::size_t x_q;
      double sim_pq = find_max(x_p, x_q, D);
      D.erase(x_q);

      double gamma = (1.0 - rho) * sim_pq;

      if (gamma <= tau) {
        break;
      }

      std::vector<std::size_t> G_k;
      G_k.push_back(x_p);
      G_k.push_back(x_q);

      while (gamma > tau) {
        std::vector<std::size_t> S;

        std::vector<double> pt(sets * sample_size, 0.0);

        std::vector<double> row;
        row.resize(sample_size * sets, 0.0);

        for (std::size_t idx : G_k) {
          std::fill(row.begin(), row.end(), 0.0);
          data->load_row(row, idx);

          for (std::size_t i = 0; i < row.size(); i++) {
            pt[i] += row[i];
          }
        }

        for (std::size_t i = 0; i < pt.size(); i++) {
          pt[i] /= G_k.size();
        }

        for (std::size_t x : D) {
          std::fill(row.begin(), row.end(), 0.0);
          data->load_row(row, x);

          double sim = 0.0;

          for (std::size_t i = 0; i < row.size(); i++) {
            sim += pt[i] * row[i];
          }

          sim /= sets;

          if (sim > gamma) {
            S.push_back(x);
          }
        }

        G_k.insert(G_k.end(), S.begin(), S.end());

        for (std::size_t idx : S) {
          D.erase(idx);
        }

        gamma *= 1.0 - rho;
      }
      clusters.push_back(G_k);
    }

    noise.insert(noise.end(), D.begin(), D.end());

    refine_step();
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

      std::unique_ptr<mass_ml::data_source_c> model_data = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_dependent_c>(random_seed, sets, sample_size, initial_data);

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

    mass_ml::parse_parameter(argc, argv, "--growth_rate", rho, "No growth rate given.");
    mass_ml::parse_parameter(argc, argv, "--threshold", tau, "No threshold given.");

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

    mass_ml::f1_measure(std::cout, *data, clusters, noise, false);
  } catch (std::exception const & ex) {
    ml_tr_handle(ex);
    std::cerr << trace::latest() << std::endl;
  }

  return EXIT_SUCCESS;
}
