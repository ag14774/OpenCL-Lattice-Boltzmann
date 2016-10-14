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
#include<sys/time.h>
#include<sys/resource.h>
//#include<fenv.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define BLOCKSIZE       16

/* struct to hold the parameter values */
typedef struct
{
  double density;       /* density per link */
  double accel;         /* density redistribution */
  double omega;         /* relaxation parameter */
  double free_cells_inv;
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */

} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  double speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr);
void preprocess_obstacles(int* obstacles,const t_param params);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
//int propagate(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr);
//int rebound(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
//int collision(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
void timestep(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles, double* avg);

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  double* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
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
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  for (unsigned int tt = 0; tt < params.maxIters; tt++)
  {
    accelerate_flow(params, cells, obstacles);
    timestep(params, &cells, &tmp_cells, obstacles, &av_vels[tt]);
    //av_vels[tt] = av_velocity(params, cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{
  /* compute weighting factors */
  double w1 = params.density * params.accel / 9.0;
  double w2 = params.density * params.accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = params.ny - 2;

  for (unsigned int jj = 0; jj < params.nx; jj++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii * params.nx + jj]
        && (cells[ii * params.nx + jj].speeds[3] - w1) > 0.0
        && (cells[ii * params.nx + jj].speeds[6] - w2) > 0.0
        && (cells[ii * params.nx + jj].speeds[7] - w2) > 0.0)
    {
      /* increase 'east-side' densities */
      cells[ii * params.nx + jj].speeds[1] += w1;
      cells[ii * params.nx + jj].speeds[5] += w2;
      cells[ii * params.nx + jj].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii * params.nx + jj].speeds[3] -= w1;
      cells[ii * params.nx + jj].speeds[6] -= w2;
      cells[ii * params.nx + jj].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

/*int propagate(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr)
{
  //t_speed* cells = *cells_ptr;
  //t_speed* tmp_cells = *tmp_cells_ptr;
  // loop over _all_ cells 
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {

      
    }
  }

  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles)
{
  //loop over the cells in the grid
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      if the cell contains an obstacle
      if (obstacles[ii * params.nx + jj])
      {

      }
    }
  }

  return EXIT_SUCCESS;
}
*/


//double sqrt13(double n)
//{
//    double result;
//
//    __asm__(
//        "fsqrt\n\t"
//        : "=t"(result) : "0"(n)
//    );
//
//    return result;
//}


void timestep(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles, double* avg)
{
  static const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  static const double twooverthree = 2.0/3.0;
  static const double two_c_sq_sq = 2.0 / 9.0;
  static const double w0 = 4.0 / 81.0 * 4.5;  /* weighting factor */
  static const double w1 = 1.0 / 9.0 * 4.5 ;  /* weighting factor */
  static const double w2 = 1.0 / 36.0 * 4.5; /* weighting factor */
  register double oneminusomega = 1.0 - params.omega;
  double tot_u = 0.0;
  t_speed* cells = *cells_ptr;
  t_speed* tmp_cells = *tmp_cells_ptr;

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    int y_n = ii+1;
    if(y_n == params.ny) y_n = 0;
    int y_s = (ii == 0) ? (params.ny - 1) : (ii - 1);
    for(unsigned int jj = 0; jj < params.nx; jj++){
        /* determine indices of axis-direction neighbours
        ** respecting periodic boundary conditions (wrap around) */
        int x_e = jj + 1;
        if (x_e == params.nx) x_e = 0;
        int x_w = (jj == 0) ? (params.nx - 1) : (jj - 1);
        /* propagate densities to neighbouring cells, following
        ** appropriate directions of travel and writing into
        ** scratch space grid */
        t_speed *const tmp_cell = &tmp_cells[ii*params.nx + jj];

        //Reverse the operation such that after each iteration the current cell is fully updated
        //and hence the loop can be merged with the next step
        if(!obstacles[ii*params.nx+jj]){
            
            double local_density = tmp_cell->speeds[0] = cells[ii * params.nx + jj].speeds[0];
            local_density += tmp_cell->speeds[1] = cells[ii * params.nx + x_w].speeds[1];
            local_density += tmp_cell->speeds[2] = cells[y_s * params.nx + jj].speeds[2];
            local_density += tmp_cell->speeds[3] = cells[ii * params.nx + x_e].speeds[3];
            local_density += tmp_cell->speeds[4] = cells[y_n * params.nx + jj].speeds[4];
            local_density += tmp_cell->speeds[5] = cells[y_s * params.nx + x_w].speeds[5];
            local_density += tmp_cell->speeds[6] = cells[y_s * params.nx + x_e].speeds[6];
            local_density += tmp_cell->speeds[7] = cells[y_n * params.nx + x_e].speeds[7];
            local_density += tmp_cell->speeds[8] = cells[y_n * params.nx + x_w].speeds[8];

            //double local_density = 0.0;
            /* compute local density total */
            //for (unsigned int kk = 0; kk < NSPEEDS; kk++)
            //{
            //    local_density += tmp_cell->speeds[kk];
            //}

            /* compute x velocity component. NO DIVISION BY LOCAL DENSITY*/
            double u_x = tmp_cell->speeds[1]
                        + tmp_cell->speeds[5]
                        + tmp_cell->speeds[8]
                        - tmp_cell->speeds[3]
                        - tmp_cell->speeds[6]
                        - tmp_cell->speeds[7];
            /* compute y velocity component. NO DIVISION BY LOCAL DENSITY */
            double u_y = tmp_cell->speeds[2]
                        + tmp_cell->speeds[5]
                        + tmp_cell->speeds[6]
                        - tmp_cell->speeds[4]
                        - tmp_cell->speeds[7]
                        - tmp_cell->speeds[8];

//EQUATIONS ARE VERY DIFFERENT BUT STILL DO THE SAME THING.
            const double u_x_sq = u_x * u_x;
            const double u_y_sq = u_y * u_y;
            const double u_xy   = u_x + u_y;
            const double u_xy2  = u_x - u_y;
            const double ld_sq  = local_density * local_density;
            const double c_sq_ld_2 = twooverthree * local_density;
            /* velocity squared */
            const double u_sq = u_x_sq + u_y_sq;
            const double ldinv = 1.0/local_density;
            const double ldinvomega = ldinv*params.omega;
            /* equilibrium densities */
            double d_equ[NSPEEDS];
            /* zero velocity density: weight w0 */
            d_equ[0] = w0 * (2*ld_sq-3*u_sq) * ldinvomega;
            /* axis speeds: weight w1 */
            d_equ[1] = w1 * ( two_c_sq_sq*ld_sq + c_sq_ld_2*u_x 
                                + u_x_sq - u_sq*c_sq ) * ldinvomega;
            d_equ[2] = w1 * ( two_c_sq_sq*ld_sq + c_sq_ld_2*u_y 
                                + u_y_sq - u_sq*c_sq ) * ldinvomega;
            d_equ[3] = w1 * ( two_c_sq_sq*ld_sq - c_sq_ld_2*u_x 
                                + u_x_sq - u_sq*c_sq ) * ldinvomega;
            d_equ[4] = w1 * ( two_c_sq_sq*ld_sq - c_sq_ld_2*u_y
                                + u_y_sq - u_sq*c_sq ) * ldinvomega;
            /* diagonal speeds: weight w2 */
            d_equ[5] = w2 * ( two_c_sq_sq*ld_sq + c_sq_ld_2*u_xy 
                                + u_xy*u_xy - u_sq*c_sq ) * ldinvomega;
            d_equ[6] = w2 * ( two_c_sq_sq*ld_sq - c_sq_ld_2*u_xy2 
                                + u_xy2*u_xy2 - u_sq*c_sq ) * ldinvomega;
            d_equ[7] = w2 * ( two_c_sq_sq*ld_sq - c_sq_ld_2*u_xy
                                + u_xy*u_xy - u_sq*c_sq ) * ldinvomega;
            d_equ[8] = w2 * ( two_c_sq_sq*ld_sq + c_sq_ld_2*u_xy2
                                + u_xy2*u_xy2 - u_sq*c_sq ) * ldinvomega;

            
            /* relaxation step */
            for (unsigned int kk = 0; kk < NSPEEDS; kk++)
            {
                tmp_cell->speeds[kk] = tmp_cell->speeds[kk]*oneminusomega;
                tmp_cell->speeds[kk] += d_equ[kk];
                //local_density += tmp_cell->speeds[kk];
            }
            tot_u += sqrt(u_x*u_x + u_y*u_y) * ldinv;

        }
        else{
            tmp_cell->speeds[0] = cells[ii * params.nx + jj].speeds[0];
            tmp_cell->speeds[3] = cells[ii * params.nx + x_w].speeds[1];
            tmp_cell->speeds[4] = cells[y_s * params.nx + jj].speeds[2];
            tmp_cell->speeds[1] = cells[ii * params.nx + x_e].speeds[3];
            tmp_cell->speeds[2] = cells[y_n * params.nx + jj].speeds[4];
            tmp_cell->speeds[7] = cells[y_s * params.nx + x_w].speeds[5];
            tmp_cell->speeds[8] = cells[y_s * params.nx + x_e].speeds[6];
            tmp_cell->speeds[5] = cells[y_n * params.nx + x_e].speeds[7];
            tmp_cell->speeds[6] = cells[y_n * params.nx + x_w].speeds[8];
        }
    }
  }
  t_speed* temp = *cells_ptr;
  *cells_ptr = *tmp_cells_ptr;
  *tmp_cells_ptr = temp;
  *avg = tot_u * params.free_cells_inv;
}

double av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  double tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;

  /* loop over all non-blocked cells */
  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
        /* local density total */
        double local_density = 0.0;

        for (unsigned int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }
        /* x-component of velocity */
        double u_x = (cells[ii * params.nx + jj].speeds[1]
                      + cells[ii * params.nx + jj].speeds[5]
                      + cells[ii * params.nx + jj].speeds[8]
                      - (cells[ii * params.nx + jj].speeds[3]
                         + cells[ii * params.nx + jj].speeds[6]
                         + cells[ii * params.nx + jj].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        double u_y = (cells[ii * params.nx + jj].speeds[2]
                      + cells[ii * params.nx + jj].speeds[5]
                      + cells[ii * params.nx + jj].speeds[6]
                      - (cells[ii * params.nx + jj].speeds[4]
                         + cells[ii * params.nx + jj].speeds[7]
                         + cells[ii * params.nx + jj].speeds[8]))
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
               int** obstacles_ptr, double** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

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

  retval = fscanf(fp, "%lf\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  int numOfFreeCells = params->nx*params->ny;

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
  *obstacles_ptr = (int*)malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  double w0 = params->density * 4.0 / 9.0;
  double w1 = params->density      / 9.0;
  double w2 = params->density      / 36.0;

  for (unsigned int ii = 0; ii < params->ny; ii++)
  {
    for (unsigned int jj = 0; jj < params->nx; jj++)
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
  for (unsigned int ii = 0; ii < params->ny; ii++)
  {
    for (unsigned int jj = 0; jj < params->nx; jj++)
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
    if(!(*obstacles_ptr)[yy * params->nx + xx])
        numOfFreeCells--;
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }
  params->free_cells_inv = 1.0/numOfFreeCells;

  /* and close the file */
  fclose(fp);

  //preprocess_obstacles(*obstacles_ptr,*params);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters);

  return EXIT_SUCCESS;
}

//reverse [begin, end]
void reverse_array(int* begin, int* end){
    while(begin<end){
        int temp = *begin;
        *begin++ = *end;
        *end-- = temp;
    }
}

void preprocess_obstacles(int* obstacles,const t_param params){
    for(unsigned int ii=0;ii<params.nx*params.ny;ii+=BLOCKSIZE){
        int obstacle_mode = 1^obstacles[ii];
        int begin_index = ii;
        int jj=ii;
        for(unsigned jj=ii;jj<ii+BLOCKSIZE;jj++){
            if(!obstacles[jj]){
                if(obstacle_mode){
                    reverse_array(&obstacles[begin_index],&obstacles[jj-1]);
                    obstacles[jj]--;
                    obstacle_mode = 0;
                    begin_index = jj;
                }
                else{
                    obstacles[jj] = obstacles[jj-1] - 1;
                }
            }
            else{
                if(obstacle_mode){
                    obstacles[jj] = obstacles[jj-1] + 1;
                }
                else{
                    reverse_array(&obstacles[begin_index],&obstacles[jj-1]);
                    obstacle_mode = 1;
                    begin_index = jj;
                }

            }
        }
        reverse_array(&obstacles[begin_index],&obstacles[jj-1]);
    }

#ifdef DEBUG
    for(unsigned int ii=0;ii<params.nx*params.ny;ii+=BLOCKSIZE){
        for(unsigned int jj=ii;jj<ii+BLOCKSIZE;jj++){
            printf("%4d ",obstacles[jj]);
        }
        printf("\n");
    }
#endif
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr)
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

  return EXIT_SUCCESS;
}


double calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells)
{
  double total = 0.0;  /* accumulator */

  for (unsigned int ii = 0; ii < params.ny; ii++)
  {
    for (unsigned int jj = 0; jj < params.nx; jj++)
    {
      for (unsigned int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii * params.nx + jj].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels)
{
  FILE* fp;                     /* file pointer */
  const double c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  double local_density;         /* per grid cell sum of densities */
  double pressure;              /* fluid pressure in grid cell */
  double u_x;                   /* x-component of velocity in grid cell */
  double u_y;                   /* y-component of velocity in grid cell */
  double u;                     /* norm--root of summed squares--of u_x and u_y */

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
      if (obstacles[ii * params.nx + jj])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0;

        for (unsigned int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii * params.nx + jj].speeds[1]
               + cells[ii * params.nx + jj].speeds[5]
               + cells[ii * params.nx + jj].speeds[8]
               - (cells[ii * params.nx + jj].speeds[3]
                  + cells[ii * params.nx + jj].speeds[6]
                  + cells[ii * params.nx + jj].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii * params.nx + jj].speeds[2]
               + cells[ii * params.nx + jj].speeds[5]
               + cells[ii * params.nx + jj].speeds[6]
               - (cells[ii * params.nx + jj].speeds[4]
                  + cells[ii * params.nx + jj].speeds[7]
                  + cells[ii * params.nx + jj].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
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
