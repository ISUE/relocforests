#pragma once

#include "node.hpp"
#include "random.hpp"
#include <vector>
#include <data.hpp>

namespace ISUE {
  namespace RelocForests {
    class Tree {
    public:
      Tree()
      {
        root = new Node();
      };

      void Train(std::vector<LabeledPixel> data, Random *random, int image_width, int image_height) 
      {
        root->feature_ = DepthAdaptiveRGB::CreateRandom(random, image_width, image_height);
      }

      // void train_recurse(Node *head, set<LabeledData*> S)
    private:
      Node *root;
    };
  }
}