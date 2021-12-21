#ifndef DS_QUEUE_H
#define DS_QUEUE_H

#include <mass_ml/ds/data_source.h>

#include <readerwriterqueue.h>

namespace mass_ml {

  class ds_queue_c : public data_source_c {
    public:
       virtual std::size_t cols() const override;
       virtual std::size_t rows() const override;

       virtual void load_row(std::vector<double> & row, std::size_t idx) const override;
       virtual void store_row(std::vector<double> const & row, std::string const & label) override;

       virtual std::string label(std::size_t idx) const override;

     protected:
       ds_queue_c(std::size_t col, std::size_t row);

       virtual void init() override;

     private:
       friend data_source_c;

       std::size_t cols_;
       std::size_t rows_;

       mutable moodycamel::BlockingReaderWriterQueue<double> queue;
  };

  inline std::size_t ds_queue_c::cols() const {
    return cols_;
  }

  inline std::size_t ds_queue_c::rows() const {
    return rows_;
  }

}

#endif // DS_QUEUE_H
