#pragma once
#include <cstdint>

namespace ISUE {
  namespace RelocForests {
    class Settings {
    public:
      // default values
      Settings() : num_trees_(5), max_tree_depth_(16), 
        num_frames_per_tree_(500), num_pixels_per_frame_(5000),
        image_width_(640), image_height_(480),
        depth_factor_(5000)
      {}

      int32_t num_trees_,
        max_tree_depth_,
        num_frames_per_tree_,
        num_pixels_per_frame_,
        image_width_,
        image_height_,
        depth_factor_; // for TUM data sets

      // intrinsics
      double fx, fy, cx, cy;

    };
  }
}
