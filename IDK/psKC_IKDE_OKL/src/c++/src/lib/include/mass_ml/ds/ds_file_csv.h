#ifndef DS_FILE_CSV_H
#define DS_FILE_CSV_H

#include <mass_ml/ds/ds_file.h>

namespace mass_ml {

  class ds_file_csv_c : public ds_file_c {
    public:
      virtual void load_row(std::vector<double> & row, std::size_t idx) const override;
      virtual void store_row(std::vector<double> const & row, std::string const & label) override;

      virtual std::string label(std::size_t idx) const override;

    protected:
      ds_file_csv_c(std::string const & dir_name, std::string const & file_name, ds_file_mode_e mode = ds_file_mode_e::read, bool no_hdr = true, bool no_class_label = false);

      virtual void init() override;

    private:
      friend data_source_c;

      bool no_hdr;
      bool no_class_label;

      std::vector<std::string> load_row(std::size_t idx) const;
  };

}

#endif // DS_FILE_CSV_H
