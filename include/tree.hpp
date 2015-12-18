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
    private:
      Node *root;
    };
  }
}