#pragma once

#include "features.hpp"

namespace ISUE {
  namespace RelocForests {
    template<class T>
    void Serialize_(std::ostream& o, const T& t)
    {
      o.write((const char*)(&t), sizeof(T));
    }

    template<class T>
    void Deserialize_(std::istream& o, T& t)
    {
      o.read((char*)(&t), sizeof(T));
    }

    class Node {
    public:
      Node()
      {
        //parent_ = nullptr;
        left_ = nullptr;
        right_ = nullptr;
      };

      ~Node()
      {
        delete left_;
        delete right_;
      }

      void Serialize(std::ostream& o) const
      {
        Serialize_(o, is_leaf_);
        Serialize_(o, is_split_);
        Serialize_(o, depth_);
        Serialize_(o, feature_);
        Serialize_(o, mode_);
      }

      void Deserialize(std::istream& i)
      {
        Deserialize_(i, is_leaf_);
        Deserialize_(i, is_split_);
        Deserialize_(i, depth_);
        Deserialize_(i, feature_);
        Deserialize_(i, mode_);
      }

      bool is_leaf_ = false;
      bool is_split_ = false;
      uint32_t depth_ = 0;
      Node *left_;
      Node *right_;
      DepthAdaptiveRGB feature_;
      Eigen::Vector3d mode_;
    };
  }
}