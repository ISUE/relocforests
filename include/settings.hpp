#pragma once

namespace ISUE {
  namespace RelocForests {
    class Settings {
    public:
      int num_trees_,
        max_tree_depth_,
        num_frames_per_tree_,
        num_pixels_per_frame_,
        image_width_,
        image_height_,
        depth_factor_;

      // intrinsics
      int fx, fy, cx, cy;

    };
  }
}