#pragma once

#include "features.hpp"

namespace ISUE {
  namespace RelocForests {
    class Node {
    public:
      Node()
      {
        left_ = nullptr;
        right_ = nullptr;
      };

      ~Node()
      {
        delete left_;
        delete right_;
        delete feature_;
      }

      bool is_leaf_;
      bool is_split_;
      Node *left_;
      Node *right_;
      DepthAdaptiveRGB *feature_;
    };
  }
}