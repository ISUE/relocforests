#pragma once

namespace ISUE {
  namespace RelocForests {
    class Data {
    public:
      Data() {};
      std::string path_to_assoc_file_;
      std::vector<std::string> depth_names_;
      std::vector<std::string> rgb_names_;
      std::vector<CameraInfo> poses_;
    };
  }
}