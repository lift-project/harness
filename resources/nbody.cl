#ifndef Tuple_float4_float4_DEFINED
#define Tuple_float4_float4_DEFINED
typedef struct {
  float4 _0;
  float4 _1;
} Tuple_float4_float4;
#endif
#ifndef Tuple_Tuple_float4_float4_float4_DEFINED
#define Tuple_Tuple_float4_float4_float4_DEFINED
typedef struct {
  Tuple_float4_float4 _0;
  float4 _1;
} Tuple_Tuple_float4_float4_float4;
#endif

#ifndef Tuple_float4_float4_DEFINED
#define Tuple_float4_float4_DEFINED
typedef struct {
  float4 _0;
  float4 _1;
} Tuple_float4_float4;
#endif

float4 id(float4 x){
  { return x; }
}
float4 calcAcc(float4 p1, float4 p2, float deltaT, float espSqr, float4 acc){
  {
  float4 r;
  r.xyz = p2.xyz - p1.xyz ;
  float distSqr = r.x*r.x + r.y*r.y + r.z*r.z;
  float invDist = 1.0f / sqrt(distSqr + espSqr);
  float invDistCube = invDist * invDist * invDist;
  float s = invDistCube * p2.w;
  float4 res;
  res.xyz = acc.xyz + s * r.xyz;
  return res;
}
 
}
Tuple_float4_float4 update(float4 pos, float4 vel, float deltaT, float4 acceleration){
  typedef Tuple_float4_float4 Tuple;
  
  {
  float4 newPos;
  newPos.xyz = pos.xyz + vel.xyz * deltaT + 0.5f * acceleration.xyz * deltaT * deltaT;
  newPos.w = pos.w;
  float4 newVel;
  newVel.xyz = vel.xyz + acceleration.xyz * deltaT;
  newVel.w = vel.w;
  Tuple t = {newPos, newVel};
  return t;
}
      
}
kernel void KERNEL(const global float* restrict v__33, const global float* restrict v__34, float v__35, float v__36, global Tuple_float4_float4* v__49, int v_N_1){ 
#ifndef WORKGROUP_GUARD
#define WORKGROUP_GUARD
#endif
WORKGROUP_GUARD
{
  /* Static local memory */
  local float v__42[512];
  /* Typed Value memory */
  float4 v__39;
  /* Private Memory */
  float4 v__40_0;
  
  for (int v_wg_id_21 = get_group_id(0);v_wg_id_21<(v_N_1 / (128));v_wg_id_21 = (v_wg_id_21 + get_num_groups(0))){
    float4 v_tmp_85 = 0.0f;
    v__39 = v_tmp_85;
    /* unroll */
    v__40_0 = id(v__39);
    /* end unroll */
    /* reduce_seq */
    for (int v_i_28 = 0;v_i_28<(v_N_1 / (128));v_i_28 = (1 + v_i_28)){
      /* iteration count is exactly 1, no loop emitted */
      {
        int v_l_id_29 = get_local_id(0);
        vstore4(id(vload4((v_l_id_29 + (128 * v_i_28)),v__33)),v_l_id_29,v__42);;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      
      /* unroll */
      /* reduce_seq */
      for (int v_i_31 = 0;v_i_31<128;v_i_31 = (1 + v_i_31)){
        v__40_0 = calcAcc(vload4(((128 * v_wg_id_21) + get_local_id(0)),v__33), vload4(v_i_31,v__42), v__36, v__35, v__40_0);
      }
      /* end reduce_seq */
      /* end unroll */
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      
    }
    /* end reduce_seq */
    /* unroll */
    v__49[((128 * v_wg_id_21) + get_local_id(0))] = update(vload4(((128 * v_wg_id_21) + get_local_id(0)),v__33), vload4(((128 * v_wg_id_21) + get_local_id(0)),v__34), v__36, v__40_0);
    /* end unroll */
  }
}}
