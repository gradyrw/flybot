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
import time

"""
Perform K rollouts of the system up to time T. This function simulates the roll
outs, and calculates the cost of each path. It returns the weighted average control
vector.
"""
def gpu_rollout(T,K,U,generator,funcs,trial, terminals_d, states_d, controls_d):
    #Get an array of random numbers from CUDA. The numbers
    #are normally distributed with mean 0 and variance
    du = generator.gen_normal(K*T, np.float32)*1
    #Compile and return gpu functions as well as 
    #arrays in constant memory
    rollout_kernel, U_d, cost_to_go, reductor, multipy = funcs
    cuda.memcpy_dtod(U_d, U.ptr, U.nbytes)
    #Set blocksize and gridsize for rollout and 
    #cost-to-go kernels
    blocksize = (T,1,1)
    gridsize = (K,1,1)

    start = time.clock()
    #Launch the kernel for simulating rollouts
    gravity = np.float32(-5)
    speed = np.float32(.5)
    rollout_kernel(controls_d, states_d, terminals_d, du, gravity, speed, block=blocksize, grid=gridsize)
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

def func1(T):
    template = """

#include <math.h>

#define T %d

__device__ __constant__ float U_d[T];
__device__ __constant__ float course_lwr[2*T];
__device__ __constant__ float course_upr[2*T];

__device__ float get_cost(float x, float y, float u) {
  float cost = 0;
  int p = round(2*x);
  if (y < course_lwr[p] || y > course_upr[p]) {
    cost += 10;
 }
  return cost;
}

__device__ float get_terminal(float x, float y) {
  return sqrt((100-x)*(100-x));
}

/**************************************************

Kernel Function for computing rollouts

***************************************************/
__global__ void rollout_kernel(float* controls_d, float* states_d, float* terminals_d, float* du, float gravity, float speed)
{

  int tdx = threadIdx.x;
  int bdx = blockIdx.x;
  float x_pos = 0;
  float y_pos = 0;
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
        if (y_pos >= 500) {
          y_pos = 500;
          y_vel = 0;
        }
        int p = round(x_pos*2);
        if (y_pos < course_lwr[p] || y_pos > course_upr[p]) {
          x_vel = 0;
          crash = 1;
        }
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

"""%T
    mod = SourceModule(template)
    func = mod.get_function("rollout_kernel")
    U_d = mod.get_global("U_d")[0]
    course_lwr = mod.get_global("course_lwr")[0]
    course_upr = mod.get_global("course_upr")[0]
    return func, U_d,course_lwr, course_upr

def func2(T):
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
  float lambda = -1.0/5000;
  states[bdx*T+tdx] = exp(lambda*s_costs[tdx]);
}

"""%T
    mod = SourceModule(template)
    func = mod.get_function("cost_to_go")
    return func

def func3():
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

def func4():
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

def obstacle_course(length, height):
    course_lwr = []
    course_upr = []
    floor = 0
    ceiling = 150
    for x in range(length):
        if (x==20):
            floor = 50
        course_lwr.append(floor)
        course_upr.append(ceiling)
    course_lwr = np.require(course_lwr, dtype=np.float32, requirements=['A','O','W','C'])
    course_upr = np.require(course_upr, dtype=np.float32, requirements=['A','O','W','C'])
    return course_lwr, course_upr

if __name__ == "__main__":
   gca()
   T = 200
   K = 10000
   U = gpuarray.zeros(T,dtype=np.float32) + 5.0
   course_lwr,course_upr = obstacle_course(200,50)
   terminals_d = gpuarray.zeros(K, dtype = np.float32)
   states_d = gpuarray.zeros(T*K, dtype=np.float32)
   controls_d = gpuarray.zeros(T*K, dtype = np.float32)
   generator = XORWOWRandomNumberGenerator()
   func1,b,c,d = func1(T)
   cuda.memcpy_htod(c,course_lwr)
   cuda.memcpy_htod(d,course_upr)
   funcs = func1,b,func2(T),func3(),func4()
   k = 0
   sum = 1000
   count = 0
   while(k < 10000):
       k += 1
       start = time.clock()
       U = gpu_rollout(T,K,U,generator,funcs,k, terminals_d, states_d, controls_d)
       U_final = U.get()
       z = np.array([[0,0],[.5,0]])
       sum = 0
       for t in range(T):
           crash = False
           if (z[0,1] < course_lwr[2*z[0,0]] or z[0,1] > course_upr[2*z[0,0]]):
               sum += 1000
               z[1,0] = 0
           z[1,1] += -5.0 + U_final[t]
           z[0,0] += z[1,0]
           z[0,1] += z[1,1]
           if (z[0,1] <= 0):
               z[0,1] = 0
               z[1,1] = 0
           if (z[0,1] >= 500):
               z[0,1] = 500
               z[1,1] = 0
       if (k % 500 == 0):
           print k
       if (sum < 10000):
           print sum
           print k
           print
       #if (k == 100):
       #   break
       if (sum == 0 or k == 1):
           count += 1
           print k
           for j in range(10):
               gca().add_patch(Rectangle((j*10,0),10,course_lwr[j*20]))
               gca().add_patch(Rectangle((j*10,course_upr[j*20]), 10, 500 - course_upr[j*20]))
           x = np.array([[0,0],[.5,0]])
           for t in range(T):
               if (x[0,1] < course_lwr[2*x[0,0]] or x[0,1] > course_upr[2*x[0,0]]):
                   x[1,0] = 0
               x[1,1] += -5.0 + U_final[t]
               x[0,0] += x[1,0]
               x[0,1] += x[1,1]
               if (x[0,1] <= 0):
                   x[0,1] = 0
                   x[1,1] = 0
               if (x[0,1] >= 500):
                   x[0,1] = 500
                   x[1,1] = 0
               plt.scatter(x[0,0],x[0,1],c='b')
           plt.show()
           if (sum == 0 and count ==10):
               break
   
