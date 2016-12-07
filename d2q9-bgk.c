/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<sys/time.h>
#include<sys/resource.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

//#include<fenv.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"
#define NUMTHREADS      1

//Vector size
#define VECSIZE 1
//#define SINGLE_WRKGRP_REDUCT


/* struct to hold the parameter values */
struct __declspec(align(32)) t_param
{
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  float free_cells_inv;
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */

};

typedef struct t_param t_param;

/* struct to hold OpenCL objects */
typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  timestep;
  cl_kernel  reduce;

  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;
  cl_mem partial_avgs;
  cl_mem avgs;

  unsigned int nwork_groups_X;
  unsigned int nwork_groups_Y;
  unsigned int work_group_size_X;
  unsigned int work_group_size_Y;

} t_ocl;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile, t_param* params, t_speed** cells_ptr,
               t_speed** tmp_cells_ptr, int** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
//int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles, t_ocl ocl, cl_mem* d_cells);
void accelerate_flow(const t_param params, cl_mem* d_cells, t_ocl ocl);

//float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells,
//               int* restrict obstacles, t_ocl ocl);

void timestep(const t_param params, cl_mem* d_cells, cl_mem* d_tmp_cells, t_ocl ocl);

void reduce(t_ocl ocl, int tt);

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr,  float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_ocl ocl;                    /* struct to hold OpenCL objects */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  cl_int err;
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  //feenableexcept(FE_INVALID | FE_OVERFLOW);
  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl);
  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // Write cells to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(t_speed) * params.nx * params.ny, cells, 0, NULL, NULL);
  checkError(err, "writing cells data", __LINE__);

  // Write obstacles to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.obstacles, CL_TRUE, 0,
    sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);

  cl_mem* d_cells_ptrs[2];
  d_cells_ptrs[0] = &ocl.cells;
  d_cells_ptrs[1] = &ocl.tmp_cells;
  unsigned char curr_read = 0;
  unsigned char curr_write = 1;

  for (unsigned int tt = 0; tt < params.maxIters;tt++)
  {

    //****1st Iteration****

    //barrier was here
    accelerate_flow(params, d_cells_ptrs[curr_read], ocl);
    //barrier was here
    
    timestep(params, d_cells_ptrs[curr_read], d_cells_ptrs[curr_write], ocl);

    reduce(ocl, tt);

    curr_read  ^= 1;
    curr_write ^= 1;
    
  }

  // Read cells from device
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(t_speed) * params.nx * params.ny, cells, 0, NULL, NULL);
  checkError(err, "reading cells data", __LINE__);

  // Read avgs from device
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.avgs, CL_TRUE, 0,
    sizeof(float) * params.maxIters, av_vels, 0, NULL, NULL);
  checkError(err, "reading cells data", __LINE__);

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, ocl));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

inline void accelerate_flow(const t_param params, cl_mem* d_cells, t_ocl ocl)
{
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), d_cells);
  checkError(err, "setting accelerate_flow arg 0", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting accelerate_flow arg 1", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_int), &params.nx);
  checkError(err, "setting accelerate_flow arg 2", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_int), &params.ny);
  checkError(err, "setting accelerate_flow arg 3", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_float), &params.density);
  checkError(err, "setting accelerate_flow arg 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_float), &params.accel);
  checkError(err, "setting accelerate_flow arg 5", __LINE__);

  // Enqueue kernel
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                               1, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for accelerate_flow kernel", __LINE__);

}


inline void timestep(const t_param params, cl_mem* d_cells, cl_mem* d_tmp_cells, t_ocl ocl)
{
    cl_int err;

    // Set kernel arguments
    err = clSetKernelArg(ocl.timestep, 0, sizeof(cl_mem), d_cells);
    checkError(err, "setting timestep arg 0",__LINE__);
    err = clSetKernelArg(ocl.timestep, 1, sizeof(cl_mem), d_tmp_cells);
    checkError(err, "setting timestep arg 1",__LINE__);
    err = clSetKernelArg(ocl.timestep, 2, sizeof(cl_mem), &ocl.obstacles);
    checkError(err, "setting timestep arg 2",__LINE__);
    err = clSetKernelArg(ocl.timestep, 3, sizeof(cl_int), &params.nx);
    checkError(err, "setting timestep arg 3",__LINE__);
    err = clSetKernelArg(ocl.timestep, 4, sizeof(cl_int), &params.ny);
    checkError(err, "setting timestep arg 4",__LINE__);
    err = clSetKernelArg(ocl.timestep, 5, sizeof(cl_float), &params.omega);
    checkError(err, "setting timestep arg 5",__LINE__);
    err = clSetKernelArg(ocl.timestep, 6, sizeof(cl_float), &params.free_cells_inv);
    checkError(err, "setting timestep arg 6",__LINE__);
    size_t work_group_size = ocl.work_group_size_X*ocl.work_group_size_Y;
    err = clSetKernelArg(ocl.timestep, 7, sizeof(float)*work_group_size, NULL); //local
    checkError(err, "setting timestep arg 7",__LINE__);
    err = clSetKernelArg(ocl.timestep, 8, sizeof(cl_mem), &ocl.partial_avgs);
    checkError(err, "setting timestep arg 8",__LINE__);

    size_t global[2] = {params.nx/VECSIZE, params.ny};
    size_t local[2] = {ocl.work_group_size_X, ocl.work_group_size_Y};

    err = clEnqueueNDRangeKernel(ocl.queue, ocl.timestep, 2, NULL, global, local, 0, NULL, NULL);
    checkError(err, "enqueueing timestep kernel", __LINE__);

    err = clFinish(ocl.queue);
    checkError(err, "waiting for timestep kernel", __LINE__);
    
}


inline void reduce(t_ocl ocl, int tt){
    cl_int err;

    // Set kernel arguments
    err = clSetKernelArg(ocl.reduce, 0, sizeof(cl_mem), &ocl.partial_avgs);
    checkError(err, "setting reduce arg 0",__LINE__);
    //err = clSetKernelArg(ocl.reduce, 1, sizeof(float), NULL);
    //checkError(err, "setting reduce arg 1",__LINE__);
    err = clSetKernelArg(ocl.reduce, 2, sizeof(cl_mem), &ocl.avgs);
    checkError(err, "setting reduce arg 2",__LINE__);
    err = clSetKernelArg(ocl.reduce, 3, sizeof(cl_int), &tt);
    checkError(err, "setting reduce arg 3",__LINE__);

    size_t global_size = ocl.nwork_groups_X*ocl.nwork_groups_Y;
    size_t global[1];
    size_t local[1];
    
    #ifdef SINGLE_WRKGRP_REDUCT

    while(global_size != 1)
    {
        global_size = global_size / 2;
        global[0] = global_size;
        if(global_size >= 32)
            local[0] = 32;
        else
            local[0] = global_size;
        global_size = global_size / local[0]; //after running the kernel
        
        err = clSetKernelArg(ocl.reduce, 1, sizeof(float)*local[0], NULL);
        checkError(err, "setting reduce arg 1",__LINE__);
    
        err = clEnqueueNDRangeKernel(ocl.queue, ocl.reduce, 1, NULL, global, local, 0, NULL, NULL);
        checkError(err, "enqueueing reduce kernel", __LINE__);

        err = clFinish(ocl.queue);
        checkError(err, "waiting for reduce kernel", __LINE__);
    }

    #else
    global_size = global_size / 2;
    global[0] = local[0] = global_size;
    
    err = clSetKernelArg(ocl.reduce, 1, sizeof(float)*local[0], NULL);
    checkError(err, "setting reduce arg 1",__LINE__);
    
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.reduce, 1, NULL, global, local, 0, NULL, NULL);
    checkError(err, "enqueueing reduce kernel", __LINE__);

    err = clFinish(ocl.queue);
    checkError(err, "waiting for reduce kernel", __LINE__);
    #endif

}

//inline float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles, t_ocl ocl)
//{
//  //static const float c_sq = 1.0 / 3.0; /* square of speed of sound */
//  static const float ic_sq = 3.0f;
//  //static const float ic_sq_sq = 9.0;
//  static const float w0 = 4.0f / 9.0f;  /* weighting factor */
//  static const float w1 = 1.0f / 9.0f;  /* weighting factor */
//  static const float w2 = 1.0f / 36.0f; /* weighting factor */
//  float tot_u = 0.0f;
// 
//  /* loop over the cells in the grid
//  ** NB the collision step is called after
//  ** the propagate step and so values of interest
//  ** are in the scratch-space grid */
//  
//  int start = 0;
//  int end   = params.ny;
//  for (unsigned int ii = start; ii < end; ii++)
//  {
//    int y_n = ii + 1;
//    if(y_n == params.ny) y_n = 0;
//    int y_s = ii - 1;
//    if(y_s == -1) y_s = params.ny-1;
//    for(unsigned int jj = 0; jj < params.nx; jj+=VECSIZE){
//        /* determine indices of axis-direction neighbours
//        ** respecting periodic boundary conditions (wrap around) */
//        float tmp[VECSIZE*NSPEEDS] __attribute__((aligned(32)));
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++){
//            int x = jj+k;
//            int x_e = x + 1;
//            if(x_e >= params.nx) x_e -= params.nx;
//            int x_w = (x == 0) ? (params.nx - 1) : (x-1);
//            tmp[VECSIZE*0+k] = cells[ii * params.nx + x].speeds[0];
//            tmp[VECSIZE*1+k] = cells[ii * params.nx + x_w].speeds[1];
//            tmp[VECSIZE*2+k] = cells[y_s * params.nx + x].speeds[2];
//            tmp[VECSIZE*3+k] = cells[ii * params.nx + x_e].speeds[3];
//            tmp[VECSIZE*4+k] = cells[y_n * params.nx + x].speeds[4];
//            tmp[VECSIZE*5+k] = cells[y_s * params.nx + x_w].speeds[5];
//            tmp[VECSIZE*6+k] = cells[y_s * params.nx + x_e].speeds[6];
//            tmp[VECSIZE*7+k] = cells[y_n * params.nx + x_e].speeds[7];
//            tmp[VECSIZE*8+k] = cells[y_n * params.nx + x_w].speeds[8];
//            
//        }
// 
//        float densvec[VECSIZE] __attribute__((aligned(32)));
// 
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++){
//            densvec[k] = tmp[VECSIZE*0+k];
//            densvec[k] += tmp[VECSIZE*1+k];
//            densvec[k] += tmp[VECSIZE*2+k];
//            densvec[k] += tmp[VECSIZE*3+k];
//            densvec[k] += tmp[VECSIZE*4+k];
//            densvec[k] += tmp[VECSIZE*5+k];
//            densvec[k] += tmp[VECSIZE*6+k];
//            densvec[k] += tmp[VECSIZE*7+k];
//            densvec[k] += tmp[VECSIZE*8+k];
//        }
// 
//        float densinv[VECSIZE] __attribute__((aligned(32)));
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++)
//        {
//            densinv[k] = 1.0f/densvec[k];
//        }
// 
//        float u_x[VECSIZE] __attribute__((aligned(32)));
//        float u_y[VECSIZE] __attribute__((aligned(32)));
// 
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++)
//        {
//            u_x[k] = tmp[VECSIZE*1+k] + tmp[VECSIZE*5+k];
//            u_x[k] += tmp[VECSIZE*8+k];
//            u_x[k] -= tmp[VECSIZE*3+k];
//            u_x[k] -= tmp[VECSIZE*6+k];
//            u_x[k] -= tmp[VECSIZE*7+k];
//            //u_x[k] *= densinv[k];
//            u_y[k] = tmp[VECSIZE*2+k] + tmp[VECSIZE*5+k];
//            u_y[k] += tmp[VECSIZE*6+k];
//            u_y[k] -= tmp[VECSIZE*4+k];
//            u_y[k] -= tmp[VECSIZE*7+k];
//            u_y[k] -= tmp[VECSIZE*8+k];
//            //u_y[k] *= densinv[k];
//        }
// 
//        float u_sq[VECSIZE] __attribute__((aligned(32)));
// 
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++)
//        {
//            u_sq[k] = u_x[k]*u_x[k] + u_y[k]*u_y[k];
//        }
// 
//        float uvec[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++)
//        {
//            uvec[VECSIZE*1+k] =   u_x[k];
//            uvec[VECSIZE*2+k] =            u_y[k];
//            uvec[VECSIZE*3+k] = - u_x[k];
//            uvec[VECSIZE*4+k] =          - u_y[k];
//            uvec[VECSIZE*5+k] =   u_x[k] + u_y[k];
//            uvec[VECSIZE*6+k] = - u_x[k] + u_y[k];
//            uvec[VECSIZE*7+k] = - u_x[k] - u_y[k];
//            uvec[VECSIZE*8+k] =   u_x[k] - u_y[k];
//        }
// 
//        float ic_sqtimesu[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++)
//        {
//            ic_sqtimesu[VECSIZE*1+k] = uvec[VECSIZE*1+k]*ic_sq;
//            ic_sqtimesu[VECSIZE*2+k] = uvec[VECSIZE*2+k]*ic_sq;
//            ic_sqtimesu[VECSIZE*3+k] = uvec[VECSIZE*3+k]*ic_sq;
//            ic_sqtimesu[VECSIZE*4+k] = uvec[VECSIZE*4+k]*ic_sq;
//            ic_sqtimesu[VECSIZE*5+k] = uvec[VECSIZE*5+k]*ic_sq;
//            ic_sqtimesu[VECSIZE*6+k] = uvec[VECSIZE*6+k]*ic_sq;
//            ic_sqtimesu[VECSIZE*7+k] = uvec[VECSIZE*7+k]*ic_sq;
//            ic_sqtimesu[VECSIZE*8+k] = uvec[VECSIZE*8+k]*ic_sq;
//        }
// 
//        float ic_sqtimesu_sq[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++)
//        {
//            ic_sqtimesu_sq[VECSIZE*1+k] = ic_sqtimesu[VECSIZE*1+k] * uvec[VECSIZE*1+k];
//            ic_sqtimesu_sq[VECSIZE*2+k] = ic_sqtimesu[VECSIZE*2+k] * uvec[VECSIZE*2+k];
//            ic_sqtimesu_sq[VECSIZE*3+k] = ic_sqtimesu[VECSIZE*3+k] * uvec[VECSIZE*3+k];
//            ic_sqtimesu_sq[VECSIZE*4+k] = ic_sqtimesu[VECSIZE*4+k] * uvec[VECSIZE*4+k];
//            ic_sqtimesu_sq[VECSIZE*5+k] = ic_sqtimesu[VECSIZE*5+k] * uvec[VECSIZE*5+k];
//            ic_sqtimesu_sq[VECSIZE*6+k] = ic_sqtimesu[VECSIZE*6+k] * uvec[VECSIZE*6+k];
//            ic_sqtimesu_sq[VECSIZE*7+k] = ic_sqtimesu[VECSIZE*7+k] * uvec[VECSIZE*7+k];
//            ic_sqtimesu_sq[VECSIZE*8+k] = ic_sqtimesu[VECSIZE*8+k] * uvec[VECSIZE*8+k];
//        }
// 
//        float d_equ[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++)
//        {
//            d_equ[VECSIZE*0+k] = w0 * (densvec[k] - 0.5f*densinv[k]*ic_sq*u_sq[k]);
//            d_equ[VECSIZE*1+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*1+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*1+k]-u_sq[k]) );
//            d_equ[VECSIZE*2+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*2+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*2+k]-u_sq[k]) );
//            d_equ[VECSIZE*3+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*3+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*3+k]-u_sq[k]) );
//            d_equ[VECSIZE*4+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*4+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*4+k]-u_sq[k]) );
//            d_equ[VECSIZE*5+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*5+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*5+k]-u_sq[k]) );
//            d_equ[VECSIZE*6+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*6+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*6+k]-u_sq[k]) );
//            d_equ[VECSIZE*7+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*7+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*7+k]-u_sq[k]) );
//            d_equ[VECSIZE*8+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*8+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*8+k]-u_sq[k]) );
//        }
// 
//        int obst=0;
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++){
//            obst+=obstacles[ii*params.nx+jj+k];
//        }
// 
//        if(!obst){
//            #pragma vector aligned
//            for(int k=0;k<VECSIZE;k++){
//                tmp_cells[ii * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + params.omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + params.omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + params.omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + params.omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + params.omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + params.omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + params.omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + params.omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + params.omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
//                tot_u += sqrt(u_sq[k]) * densinv[k];
//            }
//        }
//        else{
// 
//        #pragma vector aligned
//        for(int k=0;k<VECSIZE;k++){
//            if(!obstacles[ii * params.nx +jj +k]){
//                tmp_cells[ii * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + params.omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + params.omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + params.omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + params.omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + params.omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + params.omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + params.omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + params.omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
//                tmp_cells[ii * params.nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + params.omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
//                tot_u += sqrt(u_sq[k]) * densinv[k]; 
//            }
//            else{
//                tmp_cells[ii * params.nx + jj + k].speeds[0] = tmp[VECSIZE*0+k];
//                tmp_cells[ii * params.nx + jj + k].speeds[3] = tmp[VECSIZE*1+k];
//                tmp_cells[ii * params.nx + jj + k].speeds[4] = tmp[VECSIZE*2+k];
//                tmp_cells[ii * params.nx + jj + k].speeds[1] = tmp[VECSIZE*3+k];
//                tmp_cells[ii * params.nx + jj + k].speeds[2] = tmp[VECSIZE*4+k];
//                tmp_cells[ii * params.nx + jj + k].speeds[7] = tmp[VECSIZE*5+k];
//                tmp_cells[ii * params.nx + jj + k].speeds[8] = tmp[VECSIZE*6+k];
//                tmp_cells[ii * params.nx + jj + k].speeds[5] = tmp[VECSIZE*7+k];
//                tmp_cells[ii * params.nx + jj + k].speeds[6] = tmp[VECSIZE*8+k];
//                
//            }
//        }
//        }
//    }
//  }
// 
//  return tot_u;
//}

float av_velocity(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl)
{
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0f;

  /* loop over all non-blocked cells */
  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii*params.nx+jj])
      {
        /* local density total */
        float local_density = 0.0f;

        for (unsigned int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii*params.nx+jj].speeds[kk];
        }
        /* x-component of velocity */
        t_speed* cell = &cells[ii*params.nx+jj];
        float u_x = (cell->speeds[1]
                      + cell->speeds[5]
                      + cell->speeds[8]
                      - (cell->speeds[3]
                         + cell->speeds[6]
                         + cell->speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cell->speeds[2]
                      + cell->speeds[5]
                      + cell->speeds[6]
                      - (cell->speeds[4]
                         + cell->speeds[7]
                         + cell->speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
      }
    }
  }

  return tot_u * params.free_cells_inv;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  int free_cells = params->nx*params->ny;

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.0f / 9.0f;
  float w1 = params->density      / 9.0f;
  float w2 = params->density      / 36.0f;

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    if(!(*obstacles_ptr)[yy*params->nx+xx])
        free_cells--;
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  params->free_cells_inv = 1.0f/free_cells;

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);


  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->timestep = clCreateKernel(ocl->program, "timestep", &err);
  checkError(err, "creating timestep kernel", __LINE__);
  ocl->reduce = clCreateKernel(ocl->program, "reduce", &err);
  checkError(err, "creating reduce kernel", __LINE__);

  size_t work_group_size = 0;
  err = clGetKernelWorkGroupInfo(ocl->timestep, ocl->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL);
  checkError(err, "getting kernel work group info", __LINE__); work_group_size=params->nx;
  printf("Work group size(CL_KERNEL_WORK_GROUP_SIZE): %lu\n",work_group_size);

  //err = clGetKernelWorkGroupInfo(ocl->timestep, ocl->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &work_group_size, NULL);
  //checkError(err, "getting kernel work group info", __LINE__);
  //printf("Work group size(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE): %lu\n", work_group_size);

  ocl->work_group_size_X = params->nx / VECSIZE; //each work item will process VECSIZE cells
  ocl->work_group_size_Y = work_group_size / ocl->work_group_size_X;
  ocl->nwork_groups_X = (params->nx/VECSIZE)/ocl->work_group_size_X;
  ocl->nwork_groups_Y = params->ny / ocl->work_group_size_Y;

  printf("%dx%d workgroups with %dx%d items each\n",ocl->nwork_groups_Y, ocl->nwork_groups_X,
                                                    ocl->work_group_size_Y, ocl->work_group_size_X);

  // Allocate OpenCL buffers
  ocl->cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(t_speed) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);

  ocl->tmp_cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(t_speed) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells buffer", __LINE__);

  ocl->obstacles = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);

  ocl->partial_avgs = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * ocl->nwork_groups_X*ocl->nwork_groups_Y, NULL, &err);
  checkError(err, "creating partial_avgs buffer", __LINE__);

  ocl->avgs = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->maxIters, NULL, &err);
  checkError(err, "creating avgs buffer", __LINE__);


  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseKernel(ocl.timestep);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl)
{
  const float viscosity = 1.0f / 6.0f * (2.0f / params.omega - 1.0f);

  return av_velocity(params, cells, obstacles, ocl) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.0f;  /* accumulator */

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii * params.nx + jj].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.0f / 3.0f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii*params.nx+jj])
      {
        u_x = u_y = u = 0.0f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0f;
        t_speed* cell = &cells[ii*params.nx+jj];

        for (unsigned int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cell->speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cell->speeds[1]
               + cell->speeds[5]
               + cell->speeds[8]
               - (cell->speeds[3]
                  + cell->speeds[6]
                  + cell->speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cell->speeds[2]
               + cell->speeds[5]
               + cell->speeds[6]
               - (cell->speeds[4]
                  + cell->speeds[7]
                  + cell->speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii*params.nx+jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (unsigned int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void checkError(cl_int err, const char *op, const int line)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice()
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];
  unsigned int compute_units;

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
    printf("%2d: %s (%u compute units)\n", d, name, compute_units);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}

