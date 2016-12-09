//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS     9
#define VECSIZE     64

#define I(jj,ii,sp) ((sp)*NX*NY+(ii)*NX+(jj)) 

kernel void accelerate_flow(global float* cells,
                            global int* obstacles)
{

  /* compute weighting factors */
  const float w1 = native_divide(DENSITY * ACCEL, 9.0f);
  const float w2 = native_divide(DENSITY * ACCEL, 36.0f);

  /* modify the 2nd row of the grid */
  const int ii = NY - 2;

  /* get column index */
  int jj = get_global_id(0);

  float res1 = cells[I(jj,ii,3)];
  float res2 = cells[I(jj,ii,6)];
  float res3 = cells[I(jj,ii,7)];

  /* if the cell is not occupied and
  ** we don't send a negative density */
  int mask = obstacles[ii*NX + jj]^1;
  int mask1 = (res1-w1>0.0f) ? 1 : 0;
  int mask2 = (res2-w2>0.0f) ? 1 : 0;
  int mask3 = (res3-w2>0.0f) ? 1 : 0;
  mask = mask & mask1 & mask2 & mask3;

  /* increase 'east-side' densities */
  cells[I(jj,ii,1)] = mad( mask, w1,cells[I(jj,ii,1)] );
  cells[I(jj,ii,5)] = mad( mask, w2,cells[I(jj,ii,5)] );
  cells[I(jj,ii,8)] = mad( mask, w2,cells[I(jj,ii,8)] );
  /* decrease 'west-side' densities */
  cells[I(jj,ii,3)] = mad( mask,-w1,res1 );
  cells[I(jj,ii,6)] = mad( mask,-w2,res2 );
  cells[I(jj,ii,7)] = mad( mask,-w2,res3 );
  
}


__kernel void timestep(__global float* restrict cells,
                     __global float* restrict tmp_cells,
                     __global int* restrict obstacles, 
                     __local float* local_avgs,
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

  float tmp[NSPEEDS];
 
  int y_n = ii + 1;
  y_n = (y_n == NY) ? (0) : (y_n);
  int y_s = (ii == 0) ? (NY-1) : (ii-1);
 
  int x_e = jj + 1;
  x_e = (x_e >= NX) ? (x_e -= NX) : (x_e);
  int x_w = (jj == 0) ? (NX - 1) : (jj-1);
  
  tmp[0] = cells[ I(jj ,ii ,0) ];
  tmp[1] = cells[ I(x_w,ii ,1) ];
  tmp[2] = cells[ I(jj ,y_s,2) ];
  tmp[3] = cells[ I(x_e,ii ,3) ];
  tmp[4] = cells[ I(jj ,y_n,4) ];
  tmp[5] = cells[ I(x_w,y_s,5) ];
  tmp[6] = cells[ I(x_e,y_s,6) ];
  tmp[7] = cells[ I(x_e,y_n,7) ];
  tmp[8] = cells[ I(x_w,y_n,8) ]; 

  
  float densvec = tmp[0];
  densvec += tmp[1];
  densvec += tmp[2];
  densvec += tmp[3];
  densvec += tmp[4];
  densvec += tmp[5];
  densvec += tmp[6];
  densvec += tmp[7];
  densvec += tmp[8];
  
  float densinv = native_recip(densvec);
  

  float u_x = tmp[1] + tmp[5];
  u_x += tmp[8];
  u_x -= tmp[3];
  u_x -= tmp[6];
  u_x -= tmp[7];
  
  float u_y = tmp[2] + tmp[5];
  u_y += tmp[6];
  u_y -= tmp[4];
  u_y -= tmp[7];
  u_y -= tmp[8];
  

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
  
  int mask = obstacles[ii*NX+jj]^1;
  
  tmp_cells[I(jj,ii,lookup[0][mask])] = tmp[0] + mask*OMEGA*(d_equ[0] - tmp[0]);
  tmp_cells[I(jj,ii,lookup[1][mask])] = tmp[1] + mask*OMEGA*(d_equ[1] - tmp[1]);
  tmp_cells[I(jj,ii,lookup[2][mask])] = tmp[2] + mask*OMEGA*(d_equ[2] - tmp[2]);
  tmp_cells[I(jj,ii,lookup[3][mask])] = tmp[3] + mask*OMEGA*(d_equ[3] - tmp[3]);
  tmp_cells[I(jj,ii,lookup[4][mask])] = tmp[4] + mask*OMEGA*(d_equ[4] - tmp[4]);
  tmp_cells[I(jj,ii,lookup[5][mask])] = tmp[5] + mask*OMEGA*(d_equ[5] - tmp[5]);
  tmp_cells[I(jj,ii,lookup[6][mask])] = tmp[6] + mask*OMEGA*(d_equ[6] - tmp[6]);
  tmp_cells[I(jj,ii,lookup[7][mask])] = tmp[7] + mask*OMEGA*(d_equ[7] - tmp[7]);
  tmp_cells[I(jj,ii,lookup[8][mask])] = tmp[8] + mask*OMEGA*(d_equ[8] - tmp[8]);
  float tot_u = mask * native_sqrt(u_sq) * densinv;
 
  local_avgs[item_id] = tot_u*FREE_CELLS_INV;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  int group_id_X = get_group_id(0);
  int group_id_Y = get_group_id(1);
  int num_groups_X = get_num_groups(0);
  int num_groups_Y = get_num_groups(1);
  int groupID = mul24(group_id_Y, num_groups_X) + group_id_X;
  //if(local_size >= 128){
  //  if (item_id<64) local_avgs[item_id] += local_avgs[item_id + 64];
  //}
  for(unsigned int s=local_size/2;s>32;s>>=1){
    if(item_id<s){
        local_avgs[item_id] += local_avgs[item_id + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  //No need to synchronise in the last warp
  if(item_id < 32){
    if(local_size>=64) local_avgs[item_id] += local_avgs[item_id + 32];
    if(local_size>=32) local_avgs[item_id] += local_avgs[item_id + 16];
    if(local_size>=16) local_avgs[item_id] += local_avgs[item_id + 8];
    if(local_size>= 8) local_avgs[item_id] += local_avgs[item_id + 4];
    if(local_size>= 4) local_avgs[item_id] += local_avgs[item_id + 2];
    if(local_size>= 2) local_avgs[item_id] += local_avgs[item_id + 1];
  }
  if(item_id == 0) partial_avgs[groupID] = local_avgs[0];
 
}


kernel void reduce(global float* partial_avgs,
                   local  float* local_partial_avgs, 
                   global float* avgs, int tt)
{
    float tmp = 0;
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int num_groups = get_num_groups(0);
    int k = 2*group_id*local_size + local_id;
    local_partial_avgs[local_id] = partial_avgs[k] + partial_avgs[k+local_size];//reduce while copying from global to local
    barrier(CLK_LOCAL_MEM_FENCE);
    if(local_size >= 512){
        if(local_id<256) local_partial_avgs[local_id] += local_partial_avgs[local_id + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_size >= 256){
        if(local_id<128) local_partial_avgs[local_id] += local_partial_avgs[local_id + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_size >= 128){
        if(local_id<64) local_partial_avgs[local_id] += local_partial_avgs[local_id + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    //for(unsigned int s=local_size/2;s>0;s>>=1){
    //    if(local_id<s){
    //        local_partial_avgs[local_id] += local_partial_avgs[local_id + s];
    //    }
    //    barrier(CLK_LOCAL_MEM_FENCE);
    //}
    //No need to synchronise in the last warp
    if(local_id < 32){
        if(local_size >= 64) local_partial_avgs[local_id] += local_partial_avgs[local_id + 32];
        if(local_size >= 32) local_partial_avgs[local_id] += local_partial_avgs[local_id + 16];
        if(local_size >= 16) local_partial_avgs[local_id] += local_partial_avgs[local_id + 8];
        if(local_size >=  8) local_partial_avgs[local_id] += local_partial_avgs[local_id + 4];
        if(local_size >=  4) local_partial_avgs[local_id] += local_partial_avgs[local_id + 2];
        if(local_size >=  2) local_partial_avgs[local_id] += local_partial_avgs[local_id + 1];
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


