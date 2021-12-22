#ifndef CONFIG_H
#define CONFIG_H

#if __GNUC__ > 7 || defined(ML_IDE)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#endif // CONFIG_H
