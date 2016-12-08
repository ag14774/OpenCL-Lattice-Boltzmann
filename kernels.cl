//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS     9
#define VECSIZE     64

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
  float w1 = native_divide(density * accel, 9.0f);
  float w2 = native_divide(density * accel, 36.0f);

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  int index = mul24(ii,nx) + jj;
  int mask = obstacles[index]^1;
  int mask1 = ((cells[index].speeds[3] - w1)>0.0f) ? 1 : 0;
  int mask2 = ((cells[index].speeds[6] - w2)>0.0f) ? 1 : 0;
  int mask3 = ((cells[index].speeds[7] - w2)>0.0f) ? 1 : 0;
  mask = mask & mask1 & mask2 & mask3;

  /* increase 'east-side' densities */
  cells[index].speeds[1] = mad(mask,w1,cells[index].speeds[1]);
  cells[index].speeds[5] = mad(mask,w2,cells[index].speeds[5]);
  cells[index].speeds[8] = mad(mask,w2,cells[index].speeds[8]);
  /* decrease 'west-side' densities */
  cells[index].speeds[3] = mad(mask,-w1,cells[index].speeds[3]);
  cells[index].speeds[6] = mad(mask,-w2,cells[index].speeds[6]);
  cells[index].speeds[7] = mad(mask,-w2,cells[index].speeds[7]);
  
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
  int y_n = (ii + 1) & (ny-1);
  int x_e = (jj + 1) & (nx-1);
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
                     __local float* tmp,
                     __local volatile float* local_avgs,
                     __global float* partial_avgs) //remember to reduce partial_avg in a different kernel
{
  //static const float c_sq = 1.0 / 3.0; /* square of speed of sound */
  const float ic_sq = 3.0f;
  //static const float ic_sq_sq = 9.0;
  const float w0 = 0.4444444444444444444444f;  /* weighting factor */
  const float w1 = 0.1111111111111111111111f;  /* weighting factor */
  const float w2 = 0.0277777777777777777778f; /* weighting factor */

  const unsigned int lookup[9][2] __attribute__((aligned(32))) = {{0,0},{3,1},{4,2},{1,3},{2,4},{7,5},{8,6},{5,7},{6,8}};
 
  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  
  int ii = get_global_id(1);
  int jj = get_global_id(0);
  int local_ii = get_local_id(1);
  int local_jj = get_local_id(0);
  int local_nx = get_local_size(0);
  int local_ny = get_local_size(1);
  int local_size  = mul24(local_nx,local_ny);
  int item_id = mul24(local_ii, local_nx) + local_jj;

  //printf("y dimension:%d\n",ii);
 
  int y_n = ii + 1;
  y_n = (y_n == ny) ? (0) : (y_n);
  int y_s = (ii == 0) ? (ny-1) : (ii-1);
 
  int x_e = jj + 1;
  x_e = (x_e >= nx) ? (x_e -= nx) : (x_e);
  int x_w = (jj == 0) ? (nx - 1) : (jj-1);
  
  tmp[local_size*0+item_id] = cells[mul24(ii, nx) + jj].speeds[0];
  tmp[local_size*1+item_id] = cells[mul24(ii, nx) + x_w].speeds[1];
  tmp[local_size*2+item_id] = cells[mul24(y_s, nx) + jj].speeds[2];
  tmp[local_size*3+item_id] = cells[mul24(ii, nx) + x_e].speeds[3];
  tmp[local_size*4+item_id] = cells[mul24(y_n, nx) + jj].speeds[4];
  tmp[local_size*5+item_id] = cells[mul24(y_s, nx) + x_w].speeds[5];
  tmp[local_size*6+item_id] = cells[mul24(y_s, nx) + x_e].speeds[6];
  tmp[local_size*7+item_id] = cells[mul24(y_n, nx) + x_e].speeds[7];
  tmp[local_size*8+item_id] = cells[mul24(y_n, nx) + x_w].speeds[8]; 

  
  float densvec = tmp[local_size*0+item_id];
  densvec += tmp[local_size*1+item_id];
  densvec += tmp[local_size*2+item_id];
  densvec += tmp[local_size*3+item_id];
  densvec += tmp[local_size*4+item_id];
  densvec += tmp[local_size*5+item_id];
  densvec += tmp[local_size*6+item_id];
  densvec += tmp[local_size*7+item_id];
  densvec += tmp[local_size*8+item_id];
  
  float densinv = native_recip(densvec);
  

  float u_x = tmp[local_size*1+item_id] + tmp[local_size*5+item_id];
  u_x += tmp[local_size*8+item_id];
  u_x -= tmp[local_size*3+item_id];
  u_x -= tmp[local_size*6+item_id];
  u_x -= tmp[local_size*7+item_id];
  
  float u_y = tmp[local_size*2+item_id] + tmp[local_size*5+item_id];
  u_y += tmp[local_size*6+item_id];
  u_y -= tmp[local_size*4+item_id];
  u_y -= tmp[local_size*7+item_id];
  u_y -= tmp[local_size*8+item_id];
  

  float u_sq = u_x*u_x + u_y*u_y;
 
 
  float uvec[NSPEEDS]; //try aligning
  uvec[1] =   u_x;
  uvec[2] =         u_y;
  uvec[3] = - u_x;
  uvec[4] =       - u_y;
  uvec[5] =   u_x + u_y;
  uvec[6] = - u_x + u_y;
  uvec[7] = - u_x - u_y;
  uvec[8] =   u_x - u_y;

  
  float ic_sqtimesu[NSPEEDS];
  ic_sqtimesu[1] = uvec[1]*ic_sq;
  ic_sqtimesu[2] = uvec[2]*ic_sq;
  ic_sqtimesu[3] = uvec[3]*ic_sq;
  ic_sqtimesu[4] = uvec[4]*ic_sq;
  ic_sqtimesu[5] = uvec[5]*ic_sq;
  ic_sqtimesu[6] = uvec[6]*ic_sq;
  ic_sqtimesu[7] = uvec[7]*ic_sq;
  ic_sqtimesu[8] = uvec[8]*ic_sq;
  

  float ic_sqtimesu_sq[NSPEEDS];
  ic_sqtimesu_sq[1] = ic_sqtimesu[1] * uvec[1];
  ic_sqtimesu_sq[2] = ic_sqtimesu[2] * uvec[2];
  ic_sqtimesu_sq[3] = ic_sqtimesu[3] * uvec[3];
  ic_sqtimesu_sq[4] = ic_sqtimesu[4] * uvec[4];
  ic_sqtimesu_sq[5] = ic_sqtimesu[5] * uvec[5];
  ic_sqtimesu_sq[6] = ic_sqtimesu[6] * uvec[6];
  ic_sqtimesu_sq[7] = ic_sqtimesu[7] * uvec[7];
  ic_sqtimesu_sq[8] = ic_sqtimesu[8] * uvec[8];

  
  float d_equ[NSPEEDS];
  d_equ[0] = w0 * (densvec - 0.5f*densinv*ic_sq*u_sq);
  d_equ[1] = w1 * (densvec + ic_sqtimesu[1] + 0.5f * densinv*ic_sq * (ic_sqtimesu_sq[1]-u_sq) );
  d_equ[2] = w1 * (densvec + ic_sqtimesu[2] + 0.5f * densinv*ic_sq * (ic_sqtimesu_sq[2]-u_sq) );
  d_equ[3] = w1 * (densvec + ic_sqtimesu[3] + 0.5f * densinv*ic_sq * (ic_sqtimesu_sq[3]-u_sq) );
  d_equ[4] = w1 * (densvec + ic_sqtimesu[4] + 0.5f * densinv*ic_sq * (ic_sqtimesu_sq[4]-u_sq) );
  d_equ[5] = w2 * (densvec + ic_sqtimesu[5] + 0.5f * densinv*ic_sq * (ic_sqtimesu_sq[5]-u_sq) );
  d_equ[6] = w2 * (densvec + ic_sqtimesu[6] + 0.5f * densinv*ic_sq * (ic_sqtimesu_sq[6]-u_sq) );
  d_equ[7] = w2 * (densvec + ic_sqtimesu[7] + 0.5f * densinv*ic_sq * (ic_sqtimesu_sq[7]-u_sq) );
  d_equ[8] = w2 * (densvec + ic_sqtimesu[8] + 0.5f * densinv*ic_sq * (ic_sqtimesu_sq[8]-u_sq) );
  
  int mask = obstacles[mul24(ii,nx)+jj]^1;
  
  tmp_cells[mul24(ii,nx) + jj].speeds[lookup[0][mask]] = tmp[local_size*0+item_id] + mask*omega*(d_equ[0] - tmp[local_size*0+item_id]);
  tmp_cells[mul24(ii,nx) + jj].speeds[lookup[1][mask]] = tmp[local_size*1+item_id] + mask*omega*(d_equ[1] - tmp[local_size*1+item_id]);
  tmp_cells[mul24(ii,nx) + jj].speeds[lookup[2][mask]] = tmp[local_size*2+item_id] + mask*omega*(d_equ[2] - tmp[local_size*2+item_id]);
  tmp_cells[mul24(ii,nx) + jj].speeds[lookup[3][mask]] = tmp[local_size*3+item_id] + mask*omega*(d_equ[3] - tmp[local_size*3+item_id]);
  tmp_cells[mul24(ii,nx) + jj].speeds[lookup[4][mask]] = tmp[local_size*4+item_id] + mask*omega*(d_equ[4] - tmp[local_size*4+item_id]);
  tmp_cells[mul24(ii,nx) + jj].speeds[lookup[5][mask]] = tmp[local_size*5+item_id] + mask*omega*(d_equ[5] - tmp[local_size*5+item_id]);
  tmp_cells[mul24(ii,nx) + jj].speeds[lookup[6][mask]] = tmp[local_size*6+item_id] + mask*omega*(d_equ[6] - tmp[local_size*6+item_id]);
  tmp_cells[mul24(ii,nx) + jj].speeds[lookup[7][mask]] = tmp[local_size*7+item_id] + mask*omega*(d_equ[7] - tmp[local_size*7+item_id]);
  tmp_cells[mul24(ii,nx) + jj].speeds[lookup[8][mask]] = tmp[local_size*8+item_id] + mask*omega*(d_equ[8] - tmp[local_size*8+item_id]);
  float tot_u = mask * native_sqrt(u_sq) * densinv;
 
  local_avgs[item_id] = tot_u*free_cells_inv;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  int group_id_X = get_group_id(0);
  int group_id_Y = get_group_id(1);
  int num_groups_X = get_num_groups(0);
  int num_groups_Y = get_num_groups(1);
  int groupID = mul24(group_id_Y, num_groups_X) + group_id_X;

  for(unsigned int s=local_size/2;s>32;s>>=1){
    if(item_id<s){
        local_avgs[item_id] += local_avgs[item_id + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  //No need to synchronise in the last warp
  if(item_id < 32){
    local_avgs[item_id] += local_avgs[item_id + 32];
    local_avgs[item_id] += local_avgs[item_id + 16];
    local_avgs[item_id] += local_avgs[item_id + 8];
    local_avgs[item_id] += local_avgs[item_id + 4];
    local_avgs[item_id] += local_avgs[item_id + 2];
    local_avgs[item_id] += local_avgs[item_id + 1];
  }
  if(item_id == 0) partial_avgs[groupID] = local_avgs[0];
 
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
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 16];
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 8];
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 4];
        local_partial_avgs[local_id] += local_partial_avgs[local_id + 2];
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


