#pragma once

#include "node.hpp"

namespace ISUE {
  namespace RelocForests {
    class Tree {
    public:
      Tree()
      {
        root = new Node();
      };

      void Train() 
      {

      }

      // void train_recurse(Node *head, set<LabeledData*> S)
    private:
      Node *root;
    };
  }
}