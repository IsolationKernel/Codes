#include <mass_ml/eval/f1_measure.h>

#include <mass_ml/ds/data_source.h>

#include <algorithm>
#include <iomanip>
#include <map>

#include <cmath>

namespace {

  struct data_t {
    std::size_t value;
    std::size_t index_one;
    std::size_t index_two;

    bool operator < (data_t const & rhs) const {
      return value < rhs.value; // ascending order!
    }
  };

  double compute_precision(std::vector<std::size_t> const & confusion_matrix, std::size_t idx_cls, std::size_t idx_clu, std::size_t num_classes, std::size_t num_clusters_detected) {
    double correct = 0.0;
    double total = 0.0;

    for (std::size_t i = 0; i < num_classes; i++) {
      if (i == idx_cls) {
        correct += confusion_matrix[i * num_clusters_detected + idx_clu];
      }

      total += confusion_matrix[i * num_clusters_detected + idx_clu];
    }

    return correct / total;
  }

  double compute_recall(std::vector<std::size_t> const & confusion_matrix, std::size_t idx_cls, std::size_t idx_clu, std::size_t num_clusters_detected) {
    double correct = 0.0;
    double total = 0.0;

    for (std::size_t i = 0; i < num_clusters_detected; i++) {
      if (i == idx_clu) {
        correct += confusion_matrix[idx_cls * num_clusters_detected + i];
      }

      total += confusion_matrix[idx_cls * num_clusters_detected + i];
    }

    return correct / total;
  }

  void quick_mapping(bool noise_found, std::size_t num_clusters, std::size_t num_classes, std::vector<std::size_t> & counts, std::vector<std::size_t> & total, std::vector<double> & best) {
    std::vector<data_t> list;

    std::size_t count = noise_found ? num_clusters - 1 : num_clusters;

    for (std::size_t i = 0; i < count; i++) {
      best[i] = -1;

      for (std::size_t j = 0; j < num_classes; j++) {
        data_t data{counts[j * num_clusters + i], i, j};
        list.push_back(data);
      }
    }

    std::sort(list.begin(), list.end());
    std::reverse(list.begin(), list.end()); // descending order

    for (data_t & data : list) {
      if (data.value == 0) {
        break;
      }

      if (best[data.index_one] == -1) {
        bool found = false;

        for (std::size_t i = 0; i < num_clusters; i++) {
          if (best[i] == data.index_two) {
            found = true;
            break;
          }
        }

        if (!found) {
          best[data.index_one] = data.index_two;
        }
      }
    }
  }

}

namespace mass_ml {

  void f1_measure(std::ostream & out, data_source_c const & data, std::vector<std::vector<std::size_t>> const & clusters, std::vector<std::size_t> const & noise, bool exit_early) {
    std::size_t num_clusters_detected = clusters.size();
    std::size_t num_classes = data.label_value_count();

    if (clusters.size() == 0) {
      out << "No clusters detected!" << std::endl;
      return;
    }

    out << "# Classes       : " << num_classes << std::endl;
    out << "# Clusters found: " << num_clusters_detected << std::endl;

    bool noise_found = false;

    if (noise.size() > 0) {
      num_clusters_detected++;
      noise_found = true;
    }

    out << "Noise points    : " << (noise_found ? "yes" : "no") << std::endl;

    if (noise_found) {
      out << "# Not assigned  : " << noise.size() << std::endl;
    }

    out << std::endl;

    if (exit_early) {
      return;
    }

    typedef std::map<std::string, std::size_t> label_map_t;
    typedef std::pair<std::string, std::size_t> label_pair_t;
    label_map_t label_map;

    for (std::size_t i = 0; i < num_classes; i++) {
      label_map.insert(label_pair_t(data.label_value(i), i));
    }

    std::vector<std::size_t> confusion_matrix(num_clusters_detected * num_classes, 0);
    std::vector<std::size_t> cluster_total(num_clusters_detected, 0);

    for (std::size_t i = 0; i < num_clusters_detected; i++) {
      std::vector<std::size_t> const & next = (noise_found && (i == (num_clusters_detected - 1))) ? noise : clusters[i];

      for (std::size_t idx : next) {
#ifdef NDEBUG
        std::size_t cls_idx = label_map[data.label(idx)];
#else
        std::size_t cls_idx = label_map.at(data.label(idx));
#endif

        confusion_matrix[cls_idx * num_clusters_detected + i]++;
        cluster_total[i]++;
      }
    }

  // confustion[actual][predicted]
    out << "Confusion Matrix:" << std::endl;
    int spc_wth = 8;

    out << std::setw(spc_wth) << " ";

    for (std::size_t i = 0; i < num_clusters_detected; i++) {
      if (noise_found && (i == (num_clusters_detected - 1))) {
        out << std::setw(spc_wth) << "N";
      } else {
        out << std::setw(spc_wth) << i;
      }
    }

    out << std::setw(spc_wth) << "Tot" << std::endl;

    for (std::size_t i = 0; i < num_classes; i++) {
      out << std::setw(spc_wth) << data.label_value(i);

      std::size_t total = 0;

      for (std::size_t j = 0; j < num_clusters_detected; j++) {
        std::size_t val = confusion_matrix[i * num_clusters_detected + j];
        out << std::setw(spc_wth) << val;
        total += val;
      }

      out << std::setw(spc_wth) << total << std::endl;
    }

    {
      std::size_t total = 0;
      out << std::setw(spc_wth) << "Tot";

      for (std::size_t i = 0; i < num_clusters_detected; i++) {
        out << std::setw(spc_wth) << cluster_total[i];
        total += cluster_total[i];
      }

      out << std::setw(spc_wth) << total << std::endl;
      out << std::endl;
    }

    std::vector<double> best(num_clusters_detected + 1, -1.0);
    quick_mapping(noise_found, num_clusters_detected, num_classes, confusion_matrix, cluster_total, best);

    for (std::size_t i = 0; i < num_clusters_detected; i++) {
      out << "Cluster " << i << " <-- ";

      if (best[i] < 0) {
        out << "no class";
      } else {
        out << data.label_value(std::size_t(best[i])) << " (" << best[i] << ")";
      }

      out << std::endl;
    }

    out << std::endl;

    double total_F1_measure = 0.0;

    for (std::size_t i = 0; i < num_classes; i++) {
      bool found = false;
      std::size_t idx = 0;

      for (std::size_t j = 0; j < num_clusters_detected; j++) {
        if (best[j] == i) {
          idx = j;
          found = true;
          break;
        }
      }

      if (!found) {
        out << "Class (" << data.label_value(i) << ") F-Measure: 0.0 (not assigned)" << std::endl;
        continue;
      }

      double precision = compute_precision(confusion_matrix, std::size_t(best[idx]), idx, num_classes, num_clusters_detected);
      double recall = compute_recall(confusion_matrix, std::size_t(best[idx]), idx, num_clusters_detected);

      double f1_measure = 2.0 * precision * recall / (precision + recall);

      out << "Class (" << data.label_value(i)
          << ") F-Measure: " << f1_measure
          << " Precision: " << precision
          << " Recall: " << recall
          << std::endl;

      if (!std::isnan(f1_measure)) {
        total_F1_measure += f1_measure;
      }
    }

    out << std::endl;

    out << "Avg F1-Measure: " << (total_F1_measure / num_classes) << std::endl;
  }

}
