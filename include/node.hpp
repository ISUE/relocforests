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

      void Serialize(std::ostream& o) const
      {
        Serialize_(o, is_leaf_);
        Serialize_(o, is_split_);
        Serialize_(o, *feature_);
        Serialize_(o, modes_);
      }

      void Deserialize(std::istream& i)
      {
        Deserialize_(i, is_leaf_);
        Deserialize_(i, is_split_);
        Deserialize_(i, *feature_);
        Deserialize_(i, modes_);
      }

      bool is_leaf_ = false;
      bool is_split_ = false;
      Node *parent_;
      Node *left_;
      Node *right_;
      DepthAdaptiveRGB *feature_;
      std::vector<LabeledPixel> distribution_;
      std::vector<cv::Point3d> modes_;
    };
  }
}