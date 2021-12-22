#include <mass_ml/ds/ds_file_libsvm.h>

#include <mass_ml/tr/trace.h>

#include <algorithm>
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

      rc[':'] = std::ctype_base::space;
      rc[' '] = std::ctype_base::space;

      return &rc[0];
    }
  };

}

namespace mass_ml {

  ds_file_libsvm_c::ds_file_libsvm_c(std::string const & dir_name, std::string const & file_name, ds_file_mode_e mode) : ds_file_c(dir_name, file_name, mode) {
  }

  void ds_file_libsvm_c::init() {
    file.seekg(0, std::ios_base::beg);

    file_row_idx.clear();
    file_row_idx.push_back(0);

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
      label_values_.insert(vstrings[0]);
#else
      label_values_.insert(vstrings.at(0));
#endif

      if (vstrings.size() == 1) {
      // all attributes are zero!
        continue;
      }

#ifdef NDEBUG
      cols_ = std::max(cols_, std::size_t(std::stoull(vstrings[vstrings.size() - 2])));
#else
      cols_ = std::max(cols_, std::size_t(std::stoull(vstrings.at(vstrings.size() - 2))));
#endif
    }

    file.clear(); // to clear the EOF flags
  }

  std::string ds_file_libsvm_c::label(std::size_t idx) const {
    std::vector<std::string> vstrings = load_row(idx);

#ifdef NDEBUG
    return vstrings[0];
#else
    return vstrings.at(0);
#endif
  }

  std::vector<std::string> ds_file_libsvm_c::load_row(std::size_t idx) const {
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

  void ds_file_libsvm_c::load_row(std::vector<double> & row, std::size_t idx) const {
    std::vector<std::string> vstrings = load_row(idx);

    if (vstrings.size() == 1) {
    // all attributes are zero!
      return;
    }

    vstrings.erase(vstrings.begin(), vstrings.begin() + 1);

    for (std::size_t i = 0; i < vstrings.size() / 2; i++) {
#ifdef NDEBUG
      std::size_t idx = std::stoull(vstrings[i * 2]) - 1;
      double val = std::stod(vstrings[i * 2 + 1]);

      row[idx] = val;
#else
      std::size_t idx = std::stoull(vstrings.at(i * 2)) - 1;
      double val = std::stod(vstrings.at(i * 2 + 1));

      row.at(idx) = val;
#endif
    }
  }

  void ds_file_libsvm_c::store_row(std::vector<double> const & row, std::string const & label) {
    file << label;

    for (std::size_t i = 0; i < row.size(); i++) {
#ifdef NDEBUG
      double val = row[i];
#else
      double val = row.at(i);
#endif

      if (val != 0.0) {
        file << " " << (i + 1) << ":" << val;
      }
    }

    file << std::endl;
  }

}
