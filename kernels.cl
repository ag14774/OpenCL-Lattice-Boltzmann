#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS     9
#define VECSIZE     1

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{

  /* compute weighting factors */
  float w1 = density * accel / 9.0f;
  float w2 = density * accel / 36.0f;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0f
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0f
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0f)
  {
    /* increase 'east-side' densities */
    cells[ii * nx + jj].speeds[1] += w1;
    cells[ii * nx + jj].speeds[5] += w2;
    cells[ii * nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii * nx + jj].speeds[3] -= w1;
    cells[ii * nx + jj].speeds[6] -= w2;
    cells[ii * nx + jj].speeds[7] -= w2;
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
  /* propagate densities to neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii  * nx + jj ].speeds[0] = cells[ii * nx + jj].speeds[0]; /* central cell, no movement */
  tmp_cells[ii  * nx + x_e].speeds[1] = cells[ii * nx + jj].speeds[1]; /* east */
  tmp_cells[y_n * nx + jj ].speeds[2] = cells[ii * nx + jj].speeds[2]; /* north */
  tmp_cells[ii  * nx + x_w].speeds[3] = cells[ii * nx + jj].speeds[3]; /* west */
  tmp_cells[y_s * nx + jj ].speeds[4] = cells[ii * nx + jj].speeds[4]; /* south */
  tmp_cells[y_n * nx + x_e].speeds[5] = cells[ii * nx + jj].speeds[5]; /* north-east */
  tmp_cells[y_n * nx + x_w].speeds[6] = cells[ii * nx + jj].speeds[6]; /* north-west */
  tmp_cells[y_s * nx + x_w].speeds[7] = cells[ii * nx + jj].speeds[7]; /* south-west */
  tmp_cells[y_s * nx + x_e].speeds[8] = cells[ii * nx + jj].speeds[8]; /* south-east */
}


//__kernel void timestep(__global t_speed* restrict cells,
//                     __global t_speed* restrict tmp_cells,
//                     __global int* restrict obstacles, 
//                     int nx, int ny, float omega, float free_cells_inv,
//                     __local float* local_avgs,
//                     __global float* partial_avgs) //remember to reduce partial_avg in a different kernel
//{
//  const float c_sq         = 0.33333333333333333f; /* square of speed of sound */
//  const float twooverthree = 0.66666666666666667f;
//  const float w0           = 0.22222222222222222f;  /* weighting factor */
//  const float w1           = 0.5f ;  /* weighting factor */
//  const float w2           = 0.125f; /* weighting factor */
//  float oneminusomega      = 1.0f - omega;
//  float tot_u = 0.0;
// 
//  /* loop over the cells in the grid
//  ** NB the collision step is called after
//  ** the propagate step and so values of interest
//  ** are in the scratch-space grid */
// 
//  int ii = get_global_id(1);
//  int y_n = ii + 1;
//  if(y_n == ny) y_n = 0;
//  int y_s = ii - 1;
//  if(y_s == -1) y_s = ny-1;
//  
//  int jj = get_global_id(0) * VECSIZE;
//  
//  /* determine indices of axis-direction neighbours
//  ** respecting periodic boundary conditions (wrap around) */
//  int x_e = jj + 1;
//  if (x_e == nx) x_e = 0;
//  int x_w = (jj == 0) ? (nx - 1) : (jj - 1);
//  /* propagate densities to neighbouring cells, following
//  ** appropriate directions of travel and writing into
//  ** scratch space grid */
//  global t_speed * tmp_cell = &tmp_cells[ii*nx + jj];
//  //Reverse the operation such that after each iteration the current cell is fully updated
//  //and hence the loop can be merged with the next step
//  if(!obstacles[ii*nx+jj]){
//      float local_density = tmp_cell->speeds[0] = cells[ii * nx + jj].speeds[0]; 
//      local_density += tmp_cell->speeds[1] = cells[ii * nx + x_w].speeds[1];
//      local_density += tmp_cell->speeds[2] = cells[y_s * nx + jj].speeds[2];
//      local_density += tmp_cell->speeds[3] = cells[ii * nx + x_e].speeds[3];
//      local_density += tmp_cell->speeds[4] = cells[y_n * nx + jj].speeds[4];
//      local_density += tmp_cell->speeds[5] = cells[y_s * nx + x_w].speeds[5];
//      local_density += tmp_cell->speeds[6] = cells[y_s * nx + x_e].speeds[6];
//      local_density += tmp_cell->speeds[7] = cells[y_n * nx + x_e].speeds[7];
//      local_density += tmp_cell->speeds[8] = cells[y_n * nx + x_w].speeds[8];
//      //float local_density = 0.0;
//      /* compute local density total */
//      //for (unsigned int kk = 0; kk < NSPEEDS; kk++)
//      //{
//      //    local_density += tmp_cell->speeds[kk];
//      //}
//      /* compute x velocity component. NO DIVISION BY LOCAL DENSITY*/
//      float u_x = tmp_cell->speeds[1]
//                  + tmp_cell->speeds[5]
//                  + tmp_cell->speeds[8]
//                  - tmp_cell->speeds[3]
//                  - tmp_cell->speeds[6]
//                  - tmp_cell->speeds[7];
//      /* compute y velocity component. NO DIVISION BY LOCAL DENSITY */
//      float u_y = tmp_cell->speeds[2]
//                  + tmp_cell->speeds[5]
//                  + tmp_cell->speeds[6]
//                  - tmp_cell->speeds[4]
//                  - tmp_cell->speeds[7]
//                  - tmp_cell->speeds[8];
//  
//      const float u_x_sq = u_x * u_x;
//      const float u_y_sq = u_y * u_y;
//      const float u_xy   = u_x + u_y;
//      const float u_xy2  = u_x - u_y;
//      const float ld_sq  = local_density * local_density;
//      const float c_sq_ld_2 = twooverthree * local_density;
//      /* velocity squared */
//      const float u_sq = u_x_sq + u_y_sq;
//      const float ldinv = 1.0/local_density;
//      const float ldinvomega = ldinv*omega;
//      /* equilibrium densities */
//      float d_equ[NSPEEDS];
//      /* zero velocity density: weight w0 */
//      d_equ[0] = w0 * (2*ld_sq-3*u_sq) * ldinvomega;
//      /* axis speeds: weight w1 */
//      d_equ[1] = w1 * ( w0*ld_sq + c_sq_ld_2*u_x 
//                          + u_x_sq - u_sq*c_sq ) * ldinvomega;
//      d_equ[2] = w1 * ( w0*ld_sq + c_sq_ld_2*u_y 
//                          + u_y_sq - u_sq*c_sq ) * ldinvomega;
//      d_equ[3] = w1 * ( w0*ld_sq - c_sq_ld_2*u_x 
//                          + u_x_sq - u_sq*c_sq ) * ldinvomega;
//      d_equ[4] = w1 * ( w0*ld_sq - c_sq_ld_2*u_y
//                          + u_y_sq - u_sq*c_sq ) * ldinvomega;
//      /* diagonal speeds: weight w2 */
//      d_equ[5] = w2 * ( w0*ld_sq + c_sq_ld_2*u_xy 
//                          + u_xy*u_xy - u_sq*c_sq ) * ldinvomega;
//      d_equ[6] = w2 * ( w0*ld_sq - c_sq_ld_2*u_xy2 
//                          + u_xy2*u_xy2 - u_sq*c_sq ) * ldinvomega;
//      d_equ[7] = w2 * ( w0*ld_sq - c_sq_ld_2*u_xy
//                          + u_xy*u_xy - u_sq*c_sq ) * ldinvomega;
//      d_equ[8] = w2 * ( w0*ld_sq + c_sq_ld_2*u_xy2
//                          + u_xy2*u_xy2 - u_sq*c_sq ) * ldinvomega;
//  
//      
//      /* relaxation step */
//      for (unsigned int kk = 0; kk < NSPEEDS; kk++)
//      {
//          tmp_cell->speeds[kk] = tmp_cell->speeds[kk]*oneminusomega;
//          tmp_cell->speeds[kk] += d_equ[kk];
//          //local_density += tmp_cell->speeds[kk];
//      }
//      tot_u += sqrt(u_x*u_x + u_y*u_y) * ldinv;
//  }
//  else{
//      tmp_cell->speeds[0] = cells[ii * nx + jj].speeds[0]; 
//      tmp_cell->speeds[3] = cells[ii * nx + x_w].speeds[1];
//      tmp_cell->speeds[4] = cells[y_s * nx + jj].speeds[2];
//      tmp_cell->speeds[1] = cells[ii * nx + x_e].speeds[3];
//      tmp_cell->speeds[2] = cells[y_n * nx + jj].speeds[4];
//      tmp_cell->speeds[7] = cells[y_s * nx + x_w].speeds[5];
//      tmp_cell->speeds[8] = cells[y_s * nx + x_e].speeds[6];
//      tmp_cell->speeds[5] = cells[y_n * nx + x_e].speeds[7];
//      tmp_cell->speeds[6] = cells[y_n * nx + x_w].speeds[8];
//  }
// 
//  int local_size_X = get_local_size(0);
//  int local_size_Y = get_local_size(1); 
//  int local_id_X = get_local_id(0);
//  int local_id_Y = get_local_id(1);
//  int itemID = local_id_Y * local_size_X + local_id_X;
//  local_avgs[itemID] = tot_u*free_cells_inv;
//  barrier(CLK_LOCAL_MEM_FENCE);
//  
//  if(itemID == 0){
//    int local_size = local_size_X * local_size_Y;
//    int group_id_X = get_group_id(0);
//    int group_id_Y = get_group_id(1);
//    int num_groups_X = get_num_groups(0);
//    int num_groups_Y = get_num_groups(1);
//    int groupID = group_id_Y * num_groups_X + group_id_X;
//    partial_avgs[groupID] = 0.0f;
//    for(int l=0;l<local_size;l++) partial_avgs[groupID] += local_avgs[l];
//    
//  }
// 
// 
//}

__kernel void timestep(__global t_speed* restrict cells,
                     __global t_speed* restrict tmp_cells,
                     __global int* restrict obstacles, 
                     int nx, int ny, float omega, float free_cells_inv,
                     __local volatile float* local_avgs,
                     __global float* partial_avgs) //remember to reduce partial_avg in a different kernel
{
  //static const float c_sq = 1.0 / 3.0; /* square of speed of sound */
  const float ic_sq = 3.0f;
  //static const float ic_sq_sq = 9.0;
  const float w0 = 0.4444444444444444444444f;  /* weighting factor */
  const float w1 = 0.1111111111111111111111f;  /* weighting factor */
  const float w2 = 0.0277777777777777777778f; /* weighting factor */
  float tot_u = 0.0f;
 
  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  
  int ii = get_global_id(1);
  //printf("y dimension:%d\n",ii);
 
  int y_n = ii + 1;
  if(y_n == ny) y_n = 0;
  int y_s = ii - 1;
  if(y_s == -1) y_s = ny-1;
 
  int jj = get_global_id(0) * VECSIZE;
 
  //int start = ii*nx+jj;
  //int end = start + VECSIZE;
  //printf("Working from %d to %d\n",start,end);
 
  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  float tmp[VECSIZE*NSPEEDS] __attribute__((aligned(32)));
  #pragma vector aligned
  for(int k=0;k<VECSIZE;k++){
      int x = jj+k;
      int x_e = x + 1;
      if(x_e >= nx) x_e -= nx;
      int x_w = (x == 0) ? (nx - 1) : (x-1);
      tmp[VECSIZE*0+k] = cells[ii * nx + x].speeds[0];
      tmp[VECSIZE*1+k] = cells[ii * nx + x_w].speeds[1];
      tmp[VECSIZE*2+k] = cells[y_s * nx + x].speeds[2];
      tmp[VECSIZE*3+k] = cells[ii * nx + x_e].speeds[3];
      tmp[VECSIZE*4+k] = cells[y_n * nx + x].speeds[4];
      tmp[VECSIZE*5+k] = cells[y_s * nx + x_w].speeds[5];
      tmp[VECSIZE*6+k] = cells[y_s * nx + x_e].speeds[6];
      tmp[VECSIZE*7+k] = cells[y_n * nx + x_e].speeds[7];
      tmp[VECSIZE*8+k] = cells[y_n * nx + x_w].speeds[8];
      
  }
  
  float densvec[VECSIZE] __attribute__((aligned(32)));
  float densinv[VECSIZE] __attribute__((aligned(32)));
  
  #pragma vector aligned
  for(int k=0;k<VECSIZE;k++){
      densvec[k] = tmp[VECSIZE*0+k];
      densvec[k] += tmp[VECSIZE*1+k];
      densvec[k] += tmp[VECSIZE*2+k];
      densvec[k] += tmp[VECSIZE*3+k];
      densvec[k] += tmp[VECSIZE*4+k];
      densvec[k] += tmp[VECSIZE*5+k];
      densvec[k] += tmp[VECSIZE*6+k];
      densvec[k] += tmp[VECSIZE*7+k];
      densvec[k] += tmp[VECSIZE*8+k];
      densinv[k] = 1.0f/densvec[k];
  }
  
  float u_x[VECSIZE] __attribute__((aligned(32)));
  float u_y[VECSIZE] __attribute__((aligned(32)));
  
  #pragma vector aligned
  for(int k=0;k<VECSIZE;k++)
  {
      u_x[k] = tmp[VECSIZE*1+k] + tmp[VECSIZE*5+k];
      u_x[k] += tmp[VECSIZE*8+k];
      u_x[k] -= tmp[VECSIZE*3+k];
      u_x[k] -= tmp[VECSIZE*6+k];
      u_x[k] -= tmp[VECSIZE*7+k];
      //u_x[k] *= densinv[k];
      u_y[k] = tmp[VECSIZE*2+k] + tmp[VECSIZE*5+k];
      u_y[k] += tmp[VECSIZE*6+k];
      u_y[k] -= tmp[VECSIZE*4+k];
      u_y[k] -= tmp[VECSIZE*7+k];
      u_y[k] -= tmp[VECSIZE*8+k];
      //u_y[k] *= densinv[k];
  }
  
  float u_sq[VECSIZE] __attribute__((aligned(32)));
  
  #pragma vector aligned
  for(int k=0;k<VECSIZE;k++)
  {
      u_sq[k] = u_x[k]*u_x[k] + u_y[k]*u_y[k];
  }
  
  float uvec[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
  #pragma vector aligned
  for(int k=0;k<VECSIZE;k++)
  {
      uvec[VECSIZE*1+k] =   u_x[k];
      uvec[VECSIZE*2+k] =            u_y[k];
      uvec[VECSIZE*3+k] = - u_x[k];
      uvec[VECSIZE*4+k] =          - u_y[k];
      uvec[VECSIZE*5+k] =   u_x[k] + u_y[k];
      uvec[VECSIZE*6+k] = - u_x[k] + u_y[k];
      uvec[VECSIZE*7+k] = - u_x[k] - u_y[k];
      uvec[VECSIZE*8+k] =   u_x[k] - u_y[k];
  }
  
  float ic_sqtimesu[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
  #pragma vector aligned
  for(int k=0;k<VECSIZE;k++)
  {
      ic_sqtimesu[VECSIZE*1+k] = uvec[VECSIZE*1+k]*ic_sq;
      ic_sqtimesu[VECSIZE*2+k] = uvec[VECSIZE*2+k]*ic_sq;
      ic_sqtimesu[VECSIZE*3+k] = uvec[VECSIZE*3+k]*ic_sq;
      ic_sqtimesu[VECSIZE*4+k] = uvec[VECSIZE*4+k]*ic_sq;
      ic_sqtimesu[VECSIZE*5+k] = uvec[VECSIZE*5+k]*ic_sq;
      ic_sqtimesu[VECSIZE*6+k] = uvec[VECSIZE*6+k]*ic_sq;
      ic_sqtimesu[VECSIZE*7+k] = uvec[VECSIZE*7+k]*ic_sq;
      ic_sqtimesu[VECSIZE*8+k] = uvec[VECSIZE*8+k]*ic_sq;
  }
  
  float ic_sqtimesu_sq[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
  #pragma vector aligned
  for(int k=0;k<VECSIZE;k++)
  {
      ic_sqtimesu_sq[VECSIZE*1+k] = ic_sqtimesu[VECSIZE*1+k] * uvec[VECSIZE*1+k];
      ic_sqtimesu_sq[VECSIZE*2+k] = ic_sqtimesu[VECSIZE*2+k] * uvec[VECSIZE*2+k];
      ic_sqtimesu_sq[VECSIZE*3+k] = ic_sqtimesu[VECSIZE*3+k] * uvec[VECSIZE*3+k];
      ic_sqtimesu_sq[VECSIZE*4+k] = ic_sqtimesu[VECSIZE*4+k] * uvec[VECSIZE*4+k];
      ic_sqtimesu_sq[VECSIZE*5+k] = ic_sqtimesu[VECSIZE*5+k] * uvec[VECSIZE*5+k];
      ic_sqtimesu_sq[VECSIZE*6+k] = ic_sqtimesu[VECSIZE*6+k] * uvec[VECSIZE*6+k];
      ic_sqtimesu_sq[VECSIZE*7+k] = ic_sqtimesu[VECSIZE*7+k] * uvec[VECSIZE*7+k];
      ic_sqtimesu_sq[VECSIZE*8+k] = ic_sqtimesu[VECSIZE*8+k] * uvec[VECSIZE*8+k];
  }
  
  float d_equ[NSPEEDS*VECSIZE] __attribute__((aligned(32)));
  #pragma vector aligned
  for(int k=0;k<VECSIZE;k++)
  {
      d_equ[VECSIZE*0+k] = w0 * (densvec[k] - 0.5f*densinv[k]*ic_sq*u_sq[k]);
      d_equ[VECSIZE*1+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*1+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*1+k]-u_sq[k]) );
      d_equ[VECSIZE*2+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*2+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*2+k]-u_sq[k]) );
      d_equ[VECSIZE*3+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*3+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*3+k]-u_sq[k]) );
      d_equ[VECSIZE*4+k] = w1 * (densvec[k] + ic_sqtimesu[VECSIZE*4+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*4+k]-u_sq[k]) );
      d_equ[VECSIZE*5+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*5+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*5+k]-u_sq[k]) );
      d_equ[VECSIZE*6+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*6+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*6+k]-u_sq[k]) );
      d_equ[VECSIZE*7+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*7+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*7+k]-u_sq[k]) );
      d_equ[VECSIZE*8+k] = w2 * (densvec[k] + ic_sqtimesu[VECSIZE*8+k] + 0.5f * densinv[k]*ic_sq * (ic_sqtimesu_sq[VECSIZE*8+k]-u_sq[k]) );
  }
  
  int obst=0;
  #pragma vector aligned
  for(int k=0;k<VECSIZE;k++){
      obst+=obstacles[ii*nx+jj+k];
  }
  
  if(!obst){
      #pragma vector aligned
      for(int k=0;k<VECSIZE;k++){
          tmp_cells[ii * nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
          tmp_cells[ii * nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
          tmp_cells[ii * nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
          tmp_cells[ii * nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
          tmp_cells[ii * nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
          tmp_cells[ii * nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
          tmp_cells[ii * nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
          tmp_cells[ii * nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
          tmp_cells[ii * nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
          tot_u += sqrt(u_sq[k]) * densinv[k];
      }
  }
  else{
  
      #pragma vector aligned
      for(int k=0;k<VECSIZE;k++){
          if(!obstacles[ii * nx +jj +k]){
              tmp_cells[ii * nx + jj + k].speeds[0] = tmp[VECSIZE*0+k] + omega*(d_equ[VECSIZE*0+k] - tmp[VECSIZE*0+k]);
              tmp_cells[ii * nx + jj + k].speeds[1] = tmp[VECSIZE*1+k] + omega*(d_equ[VECSIZE*1+k] - tmp[VECSIZE*1+k]);
              tmp_cells[ii * nx + jj + k].speeds[2] = tmp[VECSIZE*2+k] + omega*(d_equ[VECSIZE*2+k] - tmp[VECSIZE*2+k]);
              tmp_cells[ii * nx + jj + k].speeds[3] = tmp[VECSIZE*3+k] + omega*(d_equ[VECSIZE*3+k] - tmp[VECSIZE*3+k]);
              tmp_cells[ii * nx + jj + k].speeds[4] = tmp[VECSIZE*4+k] + omega*(d_equ[VECSIZE*4+k] - tmp[VECSIZE*4+k]);
              tmp_cells[ii * nx + jj + k].speeds[5] = tmp[VECSIZE*5+k] + omega*(d_equ[VECSIZE*5+k] - tmp[VECSIZE*5+k]);
              tmp_cells[ii * nx + jj + k].speeds[6] = tmp[VECSIZE*6+k] + omega*(d_equ[VECSIZE*6+k] - tmp[VECSIZE*6+k]);
              tmp_cells[ii * nx + jj + k].speeds[7] = tmp[VECSIZE*7+k] + omega*(d_equ[VECSIZE*7+k] - tmp[VECSIZE*7+k]);
              tmp_cells[ii * nx + jj + k].speeds[8] = tmp[VECSIZE*8+k] + omega*(d_equ[VECSIZE*8+k] - tmp[VECSIZE*8+k]);
              tot_u += sqrt(u_sq[k]) * densinv[k]; 
          }
          else{
              tmp_cells[ii * nx + jj + k].speeds[0] = tmp[VECSIZE*0+k];
              tmp_cells[ii * nx + jj + k].speeds[3] = tmp[VECSIZE*1+k];
              tmp_cells[ii * nx + jj + k].speeds[4] = tmp[VECSIZE*2+k];
              tmp_cells[ii * nx + jj + k].speeds[1] = tmp[VECSIZE*3+k];
              tmp_cells[ii * nx + jj + k].speeds[2] = tmp[VECSIZE*4+k];
              tmp_cells[ii * nx + jj + k].speeds[7] = tmp[VECSIZE*5+k];
              tmp_cells[ii * nx + jj + k].speeds[8] = tmp[VECSIZE*6+k];
              tmp_cells[ii * nx + jj + k].speeds[5] = tmp[VECSIZE*7+k];
              tmp_cells[ii * nx + jj + k].speeds[6] = tmp[VECSIZE*8+k];
          
          }
      }
  }
 
  int local_size_X = get_local_size(0);
  int local_size_Y = get_local_size(1); 
  int local_id_X = get_local_id(0);
  int local_id_Y = get_local_id(1);
  int itemID = local_id_Y * local_size_X + local_id_X;
  local_avgs[itemID] = tot_u*free_cells_inv;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  int local_size = local_size_X * local_size_Y;
  int group_id_X = get_group_id(0);
  int group_id_Y = get_group_id(1);
  int num_groups_X = get_num_groups(0);
  int num_groups_Y = get_num_groups(1);
  int groupID = group_id_Y * num_groups_X + group_id_X;
  partial_avgs[groupID] = 0.0f;
  for(unsigned int s=local_size/2;s>32;s>>=1){
    if(itemID<s){
        local_avgs[itemID] += local_avgs[itemID + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  //No need to synchronise in the last warp
  if(itemID < 32){
    local_avgs[itemID] += local_avgs[itemID + 32];
    local_avgs[itemID] += local_avgs[itemID + 16];
    local_avgs[itemID] += local_avgs[itemID + 8];
    local_avgs[itemID] += local_avgs[itemID + 4];
    local_avgs[itemID] += local_avgs[itemID + 2];
    local_avgs[itemID] += local_avgs[itemID + 1];
  }
  if(itemID == 0) partial_avgs[groupID] = local_avgs[0];
 
}

kernel void reduce(global float* partial_avgs,
                   local volatile float* local_partial_avgs, 
                   global float* avgs, int tt)
{
    float tmp = 0;
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int num_groups = get_num_groups(0);
    int k = 2*group_id*local_size + local_id;
    int global_id = get_global_id(0);
    local_partial_avgs[local_id] = partial_avgs[k] + partial_avgs[k+local_size];//reduce while copying from global to local
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int s=local_size/2;s>32;s>>=1){
        if(local_id<s){
            local_partial_avgs[local_id] += local_partial_avgs[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //No need to synchronise in the last warp
    if(local_id < 32){
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 32];
    //    barrier(CLK_LOCAL_MEM_FENCE);
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 16];
    //    barrier(CLK_LOCAL_MEM_FENCE);
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 8];
    //    barrier(CLK_LOCAL_MEM_FENCE);
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 4];
    //    barrier(CLK_LOCAL_MEM_FENCE);
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 2];
    //    barrier(CLK_LOCAL_MEM_FENCE);
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 1];
    }

    if(local_id == 0){
        if(num_groups == 1)
            avgs[tt] = local_partial_avgs[0];
        else
            partial_avgs[group_id] = local_partial_avgs[0];
    }
    //if(gid == 0) {
    //    for(int i=0;i<size;i++){
    //        tmp += partial_avgs[i];
    //    }
    //    avgs[tt] = tmp;
    //}
}


