#include <mass_ml/ds/ds_mdl_dependent.h>
#include <mass_ml/ds/ds_mdl_independent.h>
#include <mass_ml/ds/ds_memory.h>
#include <mass_ml/fs/alg_fs_aNNE.h>

#include <algorithm>
#include <iostream>

namespace {

  std::size_t size = 100000;
  std::size_t max_size = 100000;

  std::size_t psi = 2;
  std::size_t sets = 1000;

  double mu = 0.0;
  double sigma = 1;

  double kde_h = 0.09;

  std::vector<double> data;
  std::vector<double> density;
  std::vector<double> true_density;

  void compute_RMSE() {
    std::vector<double> copy_true(true_density.begin(), true_density.begin() + size);
    double total_true = 0.0;

    for (double val : copy_true) {
      total_true += val;
    }

    for (std::size_t i = 0; i < size; i++) {
      copy_true[i] /= total_true;
    }

    double sum = 0.0;

    for (std::size_t i = 0; i < size; i++) {
      double val = copy_true[i] - density[i];
      sum += val * val;
    }

    double rmse = std::sqrt(sum / size);

    std::cout << "RMSE: " << rmse << std::endl;
  }

  void do_AcKDE() {
    std::shared_ptr<mass_ml::data_source_c> data_original = mass_ml::data_source_c::make_shared<mass_ml::ds_memory_c>(2);

    std::vector<double> row;
    row.resize(1, 0.0);

    for (std::size_t i = 0; i < size; i++) {
      row[0] = data[i];
      data_original->store_row(row, "0");
    }

    std::unique_ptr<mass_ml::alg_feature_space_c> fs_original = std::make_unique<mass_ml::alg_fs_aNNE_c>(psi, sets);
    std::unique_ptr<mass_ml::data_source_c> data_model_original = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_dependent_c>(42, sets, psi, data_original);
    std::unique_ptr<mass_ml::data_source_c> data_fs = mass_ml::data_source_c::make_unique<mass_ml::ds_memory_c>(2);
    fs_original->transform(*data_fs , *data_original, *data_model_original);

    std::vector<double> D;
    D.resize(sets * psi, 0.0);

    row.clear();
    row.resize(sets * psi, 0.0);

    for (std::size_t i = 0; i < data_fs->rows(); i++) {
      std::fill(row.begin(), row.end(), 0.0);
      data_fs->load_row(row, i);

      for (std::size_t j = 0; j < row.size(); j++) {
#ifdef NDEBUG
        D[j] += row[j];
#else
        D.at(j) += row.at(j);
#endif
      }
    }

    for (std::size_t j = 0; j < D.size(); j++) {
#ifdef NDEBUG
      D[j] /= data_fs->rows();
#else
      D.at(j) /= data_fs->rows();
#endif
    }

    double total = 0.0;

    for (std::size_t i = 0; i < data_fs->rows(); i++) {
      std::fill(row.begin(), row.end(), 0.0);
      data_fs->load_row(row, i);

      double sum = 0.0;

      for (std::size_t j = 0; j < row.size(); j++) {
#ifdef NDEBUG
        sum += D[j] * row[j];
#else
        sum += D.at(j) * row.at(j);
#endif
      }

      density.push_back(sum / sets);
      total += sum / sets;
    }

    for (std::size_t i = 0; i < density.size(); i++) {
      density[i] /= total;
    }
  }

  void do_cKDE() {
    std::shared_ptr<mass_ml::data_source_c> data_original = mass_ml::data_source_c::make_shared<mass_ml::ds_memory_c>(2);

    std::vector<double> row;
    row.resize(1, 0.0);

    for (std::size_t i = 0; i < size; i++) {
      row[0] = data[i];
      data_original->store_row(row, "0");
    }

    std::unique_ptr<mass_ml::alg_feature_space_c> fs_original = std::make_unique<mass_ml::alg_fs_aNNE_c>(psi, sets);
    std::unique_ptr<mass_ml::data_source_c> data_model_original = mass_ml::data_source_c::make_unique<mass_ml::ds_mdl_independent_c>(42, sets * psi, 2);
    std::unique_ptr<mass_ml::data_source_c> data_fs = mass_ml::data_source_c::make_unique<mass_ml::ds_memory_c>(2);
    fs_original->transform(*data_fs, *data_original, *data_model_original);

    std::vector<double> D;
    D.resize(sets * psi, 0.0);

    row.clear();
    row.resize(sets * psi, 0.0);

    for (std::size_t i = 0; i < data_fs->rows(); i++) {
      std::fill(row.begin(), row.end(), 0.0);
      data_fs->load_row(row, i);

      for (std::size_t j = 0; j < row.size(); j++) {
#ifdef NDEBUG
        D[j] += row[j];
#else
        D.at(j) += row.at(j);
#endif
      }
    }

    for (std::size_t j = 0; j < D.size(); j++) {
#ifdef NDEBUG
      D[j] /= data_fs->rows();
#else
      D.at(j) /= data_fs->rows();
#endif
    }

    double total = 0.0;

    for (std::size_t i = 0; i < data_fs->rows(); i++) {
      std::fill(row.begin(), row.end(), 0.0);
      data_fs->load_row(row, i);

      double sum = 0.0;

      for (std::size_t j = 0; j < row.size(); j++) {
#ifdef NDEBUG
        sum += D[j] * row[j];
#else
        sum += D.at(j) * row.at(j);
#endif
      }

      density.push_back(sum / sets);
      total += sum / sets;
    }

    for (std::size_t i = 0; i < density.size(); i++) {
      density[i] /= total;
    }
  }

  double std_dev(std::vector<double> const & data) {
    double mean = 0.0;
    double count = 0.0;
    double M2 = 0.0;

    for (double val : data) {
      count++;
      double delta = val - mean;
      mean += delta / count;
      M2 += delta * (val - mean);
      }

      double variance = M2 / (count - 1);
      return std::sqrt(variance);
  }

  double find_h(std::vector<double> const & data) {
    double x25 = data[std::size_t(0.25 * data.size())];
    double x75 = data[std::size_t(0.75 * data.size())];

    double R = (x75 - x25) / 1.34;
    double sigma = std_dev(data);
    double d = sigma;

    if ((R > 0) && (R < sigma)) {
      d = R;
    }

    return std::pow(1.06 * std::pow(data.size(), -0.2) * d, 2.0);
  }

  double pi () {
//    return std::atan(1.0) * 4.0;
    return std::acos(-1.0);
  }

  void do_KDE() {
    std::vector<double> copy(data.begin(), data.begin() + size);
    std::sort(copy.begin(), copy.end());

//    double h = find_h(copy);
    double h = kde_h;
    double h_sq = h * h;

    double total = 0.0;

    for (std::size_t i = 0; i < size; i++) {
      double x = data[i];

      double sum = 0.0;

      for (double y : copy) {
        double top = (x - y) / h;
        top = top * top;
        sum += std::exp(-0.5 * top) / std::sqrt(2.0 * pi());
      }

      sum /= copy.size() * h;
      density.push_back(sum);
      total += sum;
    }

    for (std::size_t i = 0; i < density.size(); i++) {
      density[i] /= total;
    }
  }

  double normpdf(double x) {
    double a = (x - mu) / sigma;
    double top = std::exp(-0.5 * a * a);
    double bottom = std::sqrt(2.0 * pi()) * sigma;

    return top / bottom;
  }

  void generate_data() {
    std::mt19937_64 mt_random{42};
    std::normal_distribution normal_dist;

    for (std::size_t i = 0; i < size; i++) {
      double x = normal_dist(mt_random);

      data.push_back(x);
      true_density.push_back(normpdf(x));
    }
  }

}

int main(int argc __attribute__((unused)), char ** argv __attribute__((unused))) {
  generate_data();

//  do_KDE();
//  do_AcKDE();
  do_cKDE();

  compute_RMSE();

  return EXIT_SUCCESS;
}
