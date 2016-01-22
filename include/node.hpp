#pragma once

#include "features.hpp"

namespace ISUE {
  namespace RelocForests {
    class Node {
    public:
      Node()
      {
        parent_ = nullptr;
        left_ = nullptr;
        right_ = nullptr;
        feature_ = nullptr;
      };

      ~Node()
      {
        delete left_;
        delete right_;
        delete feature_;
      }

      bool is_leaf_ = false;
      bool is_split_ = false;
      Node *parent_;
      Node *left_;
      Node *right_;
      DepthAdaptiveRGB *feature_;
      std::vector<LabeledPixel> distribution_;
      //Point3D mode_;
      cv::Point3d mode_;
    };
  }
}