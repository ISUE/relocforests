#pragma once
#include <cstdint>

namespace ISUE {
  namespace RelocForests {
    class Settings {
    public:
      // default values
      Settings() 
        : num_trees_(5), max_tree_depth_(16), 
        num_frames_per_tree_(500), num_pixels_per_frame_(5000),
        image_width_(640), image_height_(480),
        depth_factor_(5000),
        fx(525.0f), fy(525.0f), cx(319.5f), cy(239.5f)
      {}

      Settings(int32_t width, int32_t height, int32_t factor, double fx, double fy, double cx, double cy)
        : num_trees_(5), max_tree_depth_(16), 
        num_frames_per_tree_(500), num_pixels_per_frame_(5000),
        image_width_(width), image_height_(height),
        depth_factor_(factor),
        fx(fx), fy(fy), cx(cx), cy(cy)
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
