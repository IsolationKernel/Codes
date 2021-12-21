#ifndef AUC_H
#define AUC_H

#include <ostream>
#include <vector>

namespace mass_ml {

  class data_source_c;

// scores are ranked in descending order with anomalous points at the top
  void auc(std::ostream & out, data_source_c const & data, std::vector<double> const & scores, std::string & src_file_name, std::size_t & sample_size);

  }

#endif // AUC_H
