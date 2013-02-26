import numpy as np

class npQueue():

      def __init__(self, length, init_data):
            self.data  = np.require(init_data, requirements = ['A', 'O', 'W', 'C'])
            self.size = length
            
      def pop_add(self, value):
            for i in range(self.size - 1):
                  self.data[i] = self.data[i+1]
            self.data[self.size - 1] = value
      
      def print_data(self):
            print self.data

      
            
