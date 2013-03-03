import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from random import gauss
from random import uniform
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import *
from pycuda import gpuarray
import math
import time

"""
PI2 class for performing the PI2 algorithm in the context of tunnel navigation
 on a CUDA capable GPU. 

__init__(T,K, horiz_vel, gravity, obj_width, obj_height, obj_pos, tunnel_upr, tunnel_lwr)

-initializes a PI2 class where T is the number of timesteps to take, K is the number of rollouts
to perform for a single trial. horiz_vel is the speed at which we're moving, gravity is downward 
acceleration due to gravity, obj_width is the width of the object, obj_height is the height of the 
object, tunnel_upr an array containing information for the upper part of the tunnel and similiarly 
for tunnel lower.

rollout(U)
-performs K rollouts of the system where U is the intial starting vector

calc_path(max = 2500)

-performs trials of the system until convergence, max is the maximum number of trials 
the program will perform until it gives up.
"""
class PI2:

    def __init__(self,T, K, horiz_vel, gravity, obj_width, obj_height, 
                 init_pos,block_width, tunnel_upr, tunnel_lwr, height):
        self.init_pos = init_pos
        self.obj_width = obj_width
        self.obj_height = obj_height
        self.T = T
        self.K = K
        self.speed = horiz_vel
        self.gravity = gravity
        self.generator = XORWOWRandomNumberGenerator()
        self.terminals_d = gpuarray.zeros(K,dtype = np.float32)
        self.states_d = gpuarray.zeros(T*K, dtype=np.float32)
        self.controls_d = gpuarray.zeros(T*K, dtype=np.float32)
        self.t_upr = tunnel_upr
        self.t_lwr = tunnel_lwr
        self.height = height
        self.block_width = block_width
        end_goal = T*horiz_vel + init_pos[0]
        func,U_d,lwr,upr = self.func1(T,len(tunnel_upr), end_goal)
        cuda.memcpy_htod(upr, tunnel_upr)
        cuda.memcpy_htod(lwr, tunnel_lwr)
        self.funcs = func, U_d, self.func2(T), self.func3(), self.func4()

    """
    Perform K rollouts of the system up to time T. This function simulates the roll
    outs, and calculates the cost of each path. It returns the weighted average control
    vector.
    """
    def rollout(self,U,var):
        T = self.T
        K = self.K
        terminals_d = self.terminals_d
        states_d = self.states_d
        controls_d = self.controls_d
        #Get an array of random numbers from CUDA. The numbers
        #are normally distributed with mean 0 and variance
        du = self.generator.gen_normal(K*T, np.float32)*var
        #Compile and return gpu functions as well as 
        #arrays in constant memory
        rollout_kernel, U_d, cost_to_go, reductor, multipy = self.funcs
        cuda.memcpy_dtod(U_d, U.ptr, U.nbytes)
        #Set blocksize and gridsize for rollout and 
        #cost-to-go kernels
        blocksize = (T,1,1)
        gridsize = (K,1,1)

        start = time.clock()
        #Launch the kernel for simulating rollouts
        rollout_kernel(controls_d, states_d, terminals_d, du, 
                       np.float32(self.gravity), 
                       np.float32(self.speed),
                       np.float32(self.init_pos[0]), 
                       np.float32(self.init_pos[1]),
                       block=blocksize, grid=gridsize)
        cuda.Context.synchronize()
        #Launch the kernel for computing cost-to-go values for each state
        cost_to_go(states_d, terminals_d, block=blocksize, grid=gridsize)
        cuda.Context.synchronize()
        #Compute the normalizer, the normalizer is an array with T indices
        #which contains the sums of columns of states_d
        start = time.clock()
        j = (K-1)//16 + 1
        out_d = gpuarray.zeros(T*j, np.float32)
        gridsize = ((T-1)//16 + 1, j, 1)
        blocksize = (16, 16, 1) 
        reductor(states_d, out_d, np.int32(K), np.int32(T), grid=gridsize, block=blocksize)
        cuda.Context.synchronize()
        while (j > 1):
            _k = j
            j = (j-1)//16 + 1
            in_d = out_d
            out_d = gpuarray.zeros(T*j, np.float32)
            gridsize = ((T-1)//16 + 1, _k, 1)
            reductor(in_d, out_d, np.int32(_k), np.int32(T), grid=gridsize, block=blocksize)
            cuda.Context.synchronize()
        normalizer = out_d

        #Multiply the controls by the weighted score. The weighted score is the cost-to-go
        #function divided by the normalizer
        blocksize = (16,16,1)
        gridsize = ((T-1)//16+1, (K-1)//16 + 1,1)
        multipy(normalizer, controls_d, states_d, np.int32(T), np.int32(K), block=blocksize, grid=gridsize)
        cuda.Context.synchronize()

        #Compute the new control vector.
        j = (K-1)//16 + 1
        out_d = gpuarray.zeros(T*j, np.float32)
        gridsize = ((T-1)//16 + 1, j, 1)
        blocksize = (16, 16, 1) 
        reductor(states_d, out_d, np.int32(K), np.int32(T), grid=gridsize, block=blocksize)
        cuda.Context.synchronize()
        while (j > 1):
            _k = j
            j = (j-1)//16 + 1
            in_d = out_d
            out_d = gpuarray.zeros(T*j, np.float32)
            gridsize = ((T-1)//16 + 1, _k, 1)
            reductor(in_d, out_d, np.int32(_k), np.int32(T), grid=gridsize, block=blocksize)
            cuda.Context.synchronize()
        return out_d

    def calc_path(self, var, max = 1000, plot=False):
        print self.obj_width
        U_d = gpuarray.zeros(self.T, dtype=np.float32) + .5
        k = 0
        sum = 1000
        while(sum >= 1000 and k < max):
            k += 1
            print k
            U_d = self.rollout(U_d, var)
            U_final = U_d.get()
            z = np.array([[self.init_pos[0], self.init_pos[1]], [self.speed, 0]])
            sum = 0
            for t in range(self.T):
                crash = False
                x_pos = z[0,0]
                x_pos_floor = x_pos
                x_pos_ceil = (x_pos + self.obj_width)
                x_pc_floor = math.floor(x_pos_floor)
                x_pc_ceil = math.floor(x_pos_ceil)
                y_top = z[0,1]
                y_bottom = y_top - self.obj_height
                if (self.t_upr[x_pc_floor] < y_top):
                    crash = True
                if (self.t_upr[x_pc_ceil] < y_top):
                    crash = True
                if (y_bottom < self.t_lwr[x_pc_floor]):
                    crash = True
                if (y_bottom < self.t_lwr[x_pc_ceil]):
                    crash = True
                if (crash):
                    sum += 1000
                    z[1,0] = 0
                z[1,1] += self.gravity + U_final[t]
                z[0,0] += z[1,0]
                z[0,1] += z[1,1]
                if (z[0,1] <= 0):
                    z[0,1] = 0
                    z[1,1] = 0
                if (z[0,1] >= self.height):
                    z[0,1] = self.height
                    z[1,1] = 0
        
        print k
        if (plot):
            self.plotter(U_final)
        return U_final
    
    def plotter(self, U_final):
        gca()
        obj_width = self.obj_width
        obj_height = self.obj_height
        blocks = int(math.floor(round((self.T*self.speed/self.block_width *1.0))))
        for j in range(blocks):
            gca().add_patch(Rectangle((j*self.block_width, 0), self.block_width, self.t_lwr[j]))
            gca().add_patch(Rectangle((j*self.block_width, self.t_upr[j]), self.block_width, self.height - self.t_upr[j]))
        z = np.array([[self.init_pos[0], self.init_pos[1]], [self.speed, 0]])
        plt.scatter(z[0,0],z[0,1], c = 'b')
        for t in range(self.T):
            crash = False
            x_pos = z[0,0]
            x_pos_floor = x_pos
            x_pos_ceil = x_pos + obj_width
            x_pc_floor = math.floor(x_pos_floor)
            x_pc_ceil = math.floor(x_pos_ceil)
            y_top = z[0,1]
            y_bottom = y_top - self.obj_height
            if (self.t_upr[x_pc_floor] < y_top):
                crash = True
            if (self.t_upr[x_pc_ceil] < y_top):
                crash = True
            if (y_bottom < self.t_lwr[x_pc_floor]):
                crash = True
            if (y_bottom < self.t_lwr[x_pc_ceil]):
                crash = True
            if (crash):
                z[1,0] = 0
                z[1,1] = 0
            if (not crash):
                z[1,1] += self.gravity + U_final[t]
                if (t < 10):
                    print 5*U_final[t]
                    print 5*(-self.gravity - U_final[t])
                z[0,0] += z[1,0]
                z[0,1] += z[1,1]
            if (z[0,1] <= 0):
                z[0,1] = 0
                z[1,1] = 0
            if (z[0,1] >= self.height):
                z[0,1] = self.height
                z[1,1] = 0
            plt.scatter(z[0,0],z[0,1], c = 'b')
        plt.show()
        print


    def func1(self,T,length,end_goal):
        template = """

    #include <math.h>

    #define T %d
    #define LENGTH %d
    #define END_GOAL %d
    #define OBJ_WIDTH %d
    #define OBJ_HEIGHT %d
    #define HEIGHT %d

    __device__ __constant__ float U_d[T];
    __device__ __constant__ float course_lwr[LENGTH];
    __device__ __constant__ float course_upr[LENGTH];
 
    __device__ int test_crash(float x_pos,float y_pos) {
      int x_pc_floor = floor(x_pos);
      int x_pc_ceil = floor(x_pos + OBJ_WIDTH);
      float y_top = y_pos;
      float y_bottom = y_top - OBJ_HEIGHT;
      int crash = 0;
      if (course_upr[x_pc_floor] <= y_top) {
        crash = 1;
      }
      else if (course_upr[x_pc_ceil] <= y_top) {
        crash = 1;
      }
      else if (course_lwr[x_pc_floor] >= y_bottom) {
        crash = 1;
      }
      else if (course_lwr[x_pc_ceil] >= y_bottom) {
        crash = 1;
      }
      return crash;
   }
    __device__ float get_cost(float x, float y, float u) {
      float cost = 0;
      int crash = test_crash(x,y);
      cost = 1.0*crash;
      return cost;
    }

    __device__ float get_terminal(float x, float y) {
      return 5000*sqrt((END_GOAL - x)*(END_GOAL - x));
    }

    /**************************************************

    Kernel Function for computing rollouts

    ***************************************************/
    __global__ void rollout_kernel(float* controls_d, float* states_d, float* terminals_d, float* du, float gravity, float speed, float x_init, float y_init)
    {

      int tdx = threadIdx.x;
      int bdx = blockIdx.x;
      float x_pos = x_init;
      float y_pos = y_init;
      float x_vel = speed;
      float y_vel = 0;

      __shared__ float contr_s[T];
      __shared__ float y_pos_list[T];
      __shared__ float x_pos_list[T];
      contr_s[tdx] = du[bdx*T+tdx] + U_d[tdx];
      __syncthreads();

      if (tdx == 0) {
        int crash = 0;
        int i;
        for (i = 0; i < T; i++) {
          if (crash == 0) {
            y_vel += gravity + contr_s[i];
            x_pos += x_vel;
            y_pos += y_vel;
            if (y_pos <= 0) {
              y_pos = 0;
              y_vel = 0;
            }
            if (y_pos >= HEIGHT) {
              y_pos = HEIGHT;
              y_vel = 0;
            }
            crash = test_crash(x_pos, y_pos);
          }
          y_pos_list[i] = y_pos;
          x_pos_list[i] = x_pos;
        }  
      }
      controls_d[bdx*T + tdx] = contr_s[tdx];
      __syncthreads();

      float cost = get_cost(x_pos_list[tdx], y_pos_list[tdx], contr_s[tdx]);
      states_d[bdx*T + tdx] = cost;
      if (tdx == 0) {
        float terminal_cost = get_terminal(x_pos, y_pos);
        terminals_d[bdx] = terminal_cost;
      }
    }

    """%(T,length,end_goal,self.obj_height, self.obj_width, self.height)
        mod = SourceModule(template)
        func = mod.get_function("rollout_kernel")
        U_d = mod.get_global("U_d")[0]
        course_lwr = mod.get_global("course_lwr")[0]
        course_upr = mod.get_global("course_upr")[0]
        return func, U_d,course_lwr, course_upr

    def func2(self,T):
        template = """

    #include <math.h>

    #define T %d

    __global__ void cost_to_go(float* states, float* terminals) 
    {
      int tdx = threadIdx.x;
      int bdx = blockIdx.x;
      __shared__ float s_costs[T];
      s_costs[tdx] = states[bdx*T + tdx];
      __syncthreads();

      if (tdx == 0) {  
        float sum = terminals[bdx];
        int i;
        for (i = 0; i < T; i++) {
          sum += s_costs[T-1-i];
          s_costs[T-1-i] = sum;
        }
      }
      __syncthreads();
      float lambda = -1.0/10000;
      states[bdx*T+tdx] = exp(lambda*s_costs[tdx]);
    }

    """%T
        mod = SourceModule(template)
        func = mod.get_function("cost_to_go")
        return func

    def func3(self):
        template = """
    __global__ void reduction_kernel(float* in_d, float* out_d, int y_len, int x_len)
    {
      int tdx = threadIdx.x;
      int tdy = threadIdx.y;
      int bdx = blockIdx.x;
      int bdy = blockIdx.y;

      int x_ind = bdx*16 + tdx;
      int y_ind = bdy*16 + tdy;

      __shared__ double data[16*16];
      data[16*tdy + tdx] = 0;

      if (x_ind < x_len && y_ind < y_len) {
        data[tdy*16 + tdx] = in_d[y_ind*x_len + x_ind];
      }
      __syncthreads();

      for (int i = 8; i > 0; i>>=1) {
        if (tdy < i) {
          data[16*tdy + tdx] += data[16*(tdy+i) + tdx];
        }
        __syncthreads();
      }

      if (tdy == 0 && x_ind < x_len) {
        out_d[bdy*x_len + x_ind] = data[tdx];
      } 
    }
    """
        mod = SourceModule(template)
        return mod.get_function("reduction_kernel")

    def func4(self):
        template = """
    __global__ void multiplier(float *normalizer, float* controls, float* states_d, int x_len, int y_len)
    {
      int tdx = threadIdx.x;
      int tdy = threadIdx.y;
      int bdx = blockIdx.x;
      int bdy = blockIdx.y;

      int x_ind = 16*bdx + tdx;
      int y_ind = 16*bdy + tdy;

      float normal = normalizer[x_ind];
      if (x_ind < x_len && y_ind < y_len) {
        states_d[y_ind * x_len + x_ind] *= controls[y_ind * x_len + x_ind]/ normal;
      }
    }
    """
        mod = SourceModule(template)
        return mod.get_function("multiplier")

if __name__ == "__main__":
    T = 400
    K = 10000
    v = .05
    g = -.5
    width = .1
    height = 1
    block_width = 1
    pos = (0,50)
    tunnel_upr = np.zeros(21)
    tunnel_lwr = np.zeros(21)
    tunnel_lwr = np.require(tunnel_lwr, dtype = np.float32, requirements = ['A','O','W','C'])
    tunnel_upr = np.require(tunnel_upr, dtype = np.float32, requirements = ['A','O','W','C'])
    screen_height = 100
    tunnel_upr = tunnel_upr + 80
    tunnel_lwr = tunnel_lwr + 20
    tunnel_upr[5] = 40
    tunnel_lwr[10] = 60
    tunnel_upr[15] = 40
    tunnel_lwr[19] = 60
    gpu_comp = PI2(T,K,v,g,width,height,pos,block_width,tunnel_upr,tunnel_lwr, screen_height)
    gpu_comp.calc_path(.1,plot = True, max = 2500)
    
        


 

