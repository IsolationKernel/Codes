#ifndef DATA_SOURCE_H
#define DATA_SOURCE_H

#include <memory>
#include <string>
#include <vector>

namespace mass_ml {

/*
 * The following template codes are based from the following web-site:
 *
 * https://stackoverflow.com/questions/41517293/how-to-avoid-two-step-initialization
 *
 */

  class data_source_c {
    public:
      template <class T, class... U>
        inline static std::shared_ptr<data_source_c> make_shared(U &&... u) {
          static_assert (std::is_base_of<data_source_c, T>::value, "T must be a descendant of data_source_c");

          std::shared_ptr<data_source_c> ptr(new T(std::forward<U>(u)...)); // pre-delete add -> [](T *p) { p->destroy(); }
          ptr->init();
          return ptr;
        }

      template <class T, class... U>
        inline static std::unique_ptr<data_source_c> make_unique(U &&... u) {
          static_assert (std::is_base_of<data_source_c, T>::value, "T must be a descendant of data_source_c");

          std::unique_ptr<data_source_c> ptr(new T(std::forward<U>(u)...)); // pre-delete add -> [](T *p) { p->destroy(); }
          ptr->init();
          return ptr;
        }

      virtual ~data_source_c() = default;

      virtual std::size_t cols() const = 0;
      virtual std::size_t rows() const = 0;

      virtual void load_row(std::vector<double> & row, std::size_t idx) const = 0;
      virtual void store_row(std::vector<double> const & row, std::string const & label) = 0;

      virtual std::size_t label_value_count() const = 0;
      virtual std::string label_value(std::size_t idx) const = 0;
      virtual std::string label(std::size_t idx) const = 0;

    protected:
      virtual void init() = 0;
  };

}

#endif // DATA_SOURCE_H
