#include <mass_ml/ds/ds_tmp_file.h>

#include <mass_ml/tr/trace.h>

#include <mass_ml/config_fs.h>

#include <iostream>

#include <cerrno>
#include <cstring>

#include <unistd.h>

namespace {

  char * tmp_storage_path = nullptr;

}

namespace mass_ml {

  ds_tmp_file_c::ds_tmp_file_c(std::size_t col) : col_(col) {
  }

  ds_tmp_file_c::~ds_tmp_file_c() {
    if (temp_file != nullptr) {
      fclose(temp_file);
      unlink(file_name);
      free(file_name);
    }
  }

  void ds_tmp_file_c::init() {
#ifdef BRK_WIN
//    char tmp [] = "./test.XXXXXX";
//    int fd = mkstemp(tmp);
//    temp_file = fdopen(fd, "w+");

//    if (fd == -1) {
//      ml_tr_initiate("Windows is even more broken!!!");
//    }

//    unlink(tmp);

    ml_tr_initiate("Windows is not supported for temp files.");
#else
//    temp_file = std::tmpfile(); // note that it is automatically deleted when program exited.

    if (tmp_storage_path == nullptr) {
      ml_tr_initiate("'init_tmp_dir' has not been called.");
    }

    file_name = strdup(tmp_storage_path);

    int fd = mkstemp(file_name);

    if (fd == -1) {
      ml_tr_initiate("Could not create a temp file.");
    }

    temp_file = fdopen(fd, "w+");

    if (temp_file == nullptr) {
      std::cerr << std::strerror(errno) << std::endl;
      ml_tr_initiate("Could not create a temp file.");
    }
#endif
  }

  std::string ds_tmp_file_c::label(std::size_t idx __attribute__((unused))) const {
#ifdef NDEBUG
    return labels[idx];
#else
    return labels.at(idx);
#endif
  }

  void ds_tmp_file_c::load_row(std::vector<double> & row, std::size_t idx) const {
    std::fseek(temp_file, idx * sizeof(double) * col_, SEEK_SET);
    std::fread(row.data(), sizeof(double), col_, temp_file);
  }

  std::size_t ds_tmp_file_c::rows() const {
    std::fseek(temp_file, 0, SEEK_END);
    return std::ftell(temp_file) / sizeof (double) / col_;
  }

  void ds_tmp_file_c::store_row(std::vector<double> const & row, std::string const & label __attribute__((unused))) {
    labels.push_back(label);
    label_values_.insert(label);

    std::fwrite(reinterpret_cast<char const *>(&row[0]), sizeof(double), col_, temp_file);
  }

  void clean_up_tmp_dir() {
    if (tmp_storage_path != nullptr) {
      free(tmp_storage_path);
      tmp_storage_path = nullptr;
    }
  }

  void init_tmp_dir(std::string const & tmp_dir) {
    std::stringstream is;
    is << tmp_dir << fs::path::preferred_separator << "tmp_stg_XXXXXX";

    tmp_storage_path = strdup(is.str().c_str());

    if (tmp_storage_path == nullptr) {
      ml_tr_initiate("Could not allocate memory for temp name storage.");
    }
  }

}
