#include <mass_ml/ds/ds_file.h>
#include <mass_ml/tr/trace.h>

#include <mass_ml/config_fs.h>

namespace mass_ml {

  ds_file_c::ds_file_c(std::string const & dir_name, std::string const & file_name, ds_file_mode_e mode) : dir_name_(dir_name), file_name_(file_name) {
    std::ios_base::openmode om =
      std::ios_base::binary | (mode == ds_file_mode_e::read ? std::ios_base::in : std::ios_base::out);

    std::stringstream is;
    is << dir_name << fs::path::preferred_separator << file_name;
    file.open(is.str(), om);

    if (!file.is_open()) {
      ml_tr_initiate("Cannot open file: " + file_name);
    }
  }

}
