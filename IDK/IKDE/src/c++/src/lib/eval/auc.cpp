#include <mass_ml/eval/auc.h>

#include <mass_ml/ds/data_source.h>

#include <algorithm>
#include <map>
#include <numeric>

#include <fstream>
#include <sstream>

namespace mass_ml {

  void auc(std::ostream & out, data_source_c const & data, std::vector<double> const & scores, std::string & src_file_name, std::size_t & sample_size) {
    std::vector<std::size_t> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
      [scores](std::size_t i1, std::size_t i2) {return scores[i1] > scores[i2];}); // descending

// find the anomaly label by assuming the anomalous point are in the minority
    typedef std::pair<std::string const, std::size_t> pair_t;
    std::map<std::string, std::size_t> labels;

    for (std::size_t i = 0; i < data.rows(); i++) {
      std::string label = data.label(i);

      std::map<std::string, std::size_t>::iterator it = labels.find(label);

      if (it == labels.end()) {
        labels[label] = 0;
      } else {
        it->second++;
      }
    }

    std::size_t best_score = std::numeric_limits<std::size_t>::max();
    std::string anomaly_label = "";

    for (pair_t const & entry : labels) {
      if (entry.second < best_score) {
        best_score = entry.second;
        anomaly_label = entry.first;
      }
    }


// now compute the AUC
    double true_positive = 0.0;
    double false_positive = 0.0;
    double sum = 0.0;

    for (std::size_t i = 0; i < data.rows(); i++) {
      std::size_t id = idx[i];
      std::string label = data.label(id);

      if (anomaly_label == label) {
        true_positive++;
      } else {
        false_positive++;
        sum += true_positive;
      }
    }

    double auc = sum / (true_positive * false_positive);
	std::ofstream output;
    std::stringstream is;
    is << "auc.csv";
    output.open(is.str(),std::ios::app);
    //output << std::setprecision(std::numeric_limits<double>::max_digits10 + 2);
    output << src_file_name << "," << sample_size << "," << auc << std::endl;
    out << "AUC: "  << auc << std::endl;
  }

}
