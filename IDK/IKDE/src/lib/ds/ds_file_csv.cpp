#include <mass_ml/ds/ds_file_csv.h>

#include <mass_ml/tr/trace.h>

#include <iomanip>
#include <iterator>
#include <sstream>

#include <cstring>

namespace {

// https://stackoverflow.com/questions/5607589/right-way-to-split-an-stdstring-into-a-vectorstring
  struct tokens: std::ctype<char> {
    tokens(): std::ctype<char>(get_table()) {}

    static std::ctype_base::mask const * get_table() {
      typedef std::ctype<char> cctype;
      static cctype::mask const * const_rc= cctype::classic_table();

      static cctype::mask rc[cctype::table_size];
      std::memcpy(rc, const_rc, cctype::table_size * sizeof(cctype::mask));

      rc[','] = std::ctype_base::space;
      rc[' '] = std::ctype_base::space;
      rc['\t'] = std::ctype_base::space;

      return &rc[0];
    }
  };

}

namespace mass_ml {

  ds_file_csv_c::ds_file_csv_c(std::string const & dir_name, std::string const & file_name, ds_file_mode_e mode, bool no_hdr_, bool no_class_label_) : ds_file_c(dir_name, file_name, mode), no_hdr(no_hdr_), no_class_label(no_class_label_) {
  }

  void ds_file_csv_c::init() {
    file.seekg(0, std::ios_base::beg);

    file_row_idx.clear();

    if (no_hdr) {
      file_row_idx.push_back(0);
    } else {
    // skip the header row
      std::string line;
      std::getline(file, line, '\n');
      file_row_idx.push_back(file.tellg());
    }

    for (std::string line; std::getline(file, line, '\n'); ) {
      file_row_idx.push_back(file.tellg());

      if (line.empty()) {
        continue;
      }

      std::istringstream ss(line);
      ss.imbue(std::locale(std::locale(), new tokens()));
      std::istream_iterator<std::string> begin(ss);
      std::istream_iterator<std::string> end;
      std::vector<std::string> vstrings(begin, end);

#ifdef NDEBUG
      cols_ = std::max(cols_, vstrings.size());
#else
      cols_ = std::max(cols_, vstrings.size());
#endif

      if (!no_class_label) {
#ifdef NDEBUG
        label_values_.insert(vstrings[vstrings.size() - 1]);
#else
        label_values_.insert(vstrings.at(vstrings.size() - 1));
#endif
      }
    }

    if (!no_class_label) {
      cols_--; // don't count the class label
    }

    file.clear(); // to clear the EOF flags
  }

  std::string ds_file_csv_c::label(std::size_t idx __attribute__((unused))) const {
    std::vector<std::string> vstrings = load_row(idx);

#ifdef NDEBUG
    return vstrings[vstrings.size() - 1];
#else
    return vstrings.at(vstrings.size() - 1);
#endif
  }

  std::vector<std::string> ds_file_csv_c::load_row(std::size_t idx) const {
#ifdef NDEBUG
    file.seekg(file_row_idx[idx], std::ios_base::beg);
#else
    file.seekg(file_row_idx.at(idx), std::ios_base::beg);
#endif

    std::string line;
    std::getline(file, line);

    file.clear(); // to clear the EOF flags

    if (line.empty()) {
        return std::vector<std::string>();
    }

    std::istringstream ss(line);
    ss.imbue(std::locale(std::locale(), new tokens()));
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    return std::vector<std::string>(begin, end);
  }

  void ds_file_csv_c::load_row(std::vector<double> &row, std::size_t idx) const {
    std::vector<std::string> vstrings = load_row(idx);

    std::size_t last = no_class_label ? vstrings.size() : vstrings.size() - 1;

    for (std::size_t i = 0; i < last; i++) {
#ifdef NDEBUG
      double val = std::stod(vstrings[i]);
      row[i] = val;
#else
      double val = std::stod(vstrings.at(i));
      row.at(i) = val;
#endif
    }
  }

  void ds_file_csv_c::store_row(std::vector<double> const & row, std::string const & label) {
    file << std::setprecision(std::numeric_limits<double>::max_digits10 + 2);

    for (std::size_t i = 0; i < row.size(); i++) {
  #ifdef NDEBUG
      file << row[i] << ",";
  #else
      file << row.at(i) << ",";
  #endif
    }

    file << label << std::endl;
  }

}
