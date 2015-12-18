#pragma once

#include "data.hpp"
#include "settings.hpp"
#include "tree.hpp"

#include <vector>

namespace ISUE {
  namespace RelocForests {
    class Forest {
    public:
      Forest(Data *data, Settings *settings) 
        : data_(data), settings_(settings)
      {
        for (int i = 0; i < settings->num_trees_; ++i)
          forest_.push_back(new Tree());
      };

      void Train()
      {
        // train each tree individually 
        for (auto t : forest_) {
          // randomly choose frames
          // randomly sample pixels per frame
          // get labels at pixel (x,y,z)
          // train tree with set of pixel recursively
          t->Train();
        }
      }

    private:
      Data *data_;
      Settings *settings_;
      std::vector<Tree*> forest_;
    };
  }

}