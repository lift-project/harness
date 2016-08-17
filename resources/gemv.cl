#ifndef Tuple_float_float_DEFINED
#define Tuple_float_float_DEFINED
typedef struct {
  float _0;
  float _1;
} Tuple_float_float;
#endif

float multAndSumUp(float acc, float l, float r){
  { return acc + (l * r); }
}
float id(float x){
  { return x; }
}
float mult(float l, float r){
  { return l * r; }
}
kernel void KERNEL(const global float* restrict v__40, const global float* restrict v__41, const global float* restrict v__42, float v__43, float v__44, global float* v__57, int v_M_1, int v_N_0){ 
#ifndef WORKGROUP_GUARD
#define WORKGROUP_GUARD
#endif
WORKGROUP_GUARD
{
  /* Static local memory */
  /* Typed Value memory */
  float v__47;
  /* Private Memory */
  float v__52_0;
  
  float v__54_0;
  
  float v__56_0;
  
  for (int v_gl_id_34 = get_global_id(0);v_gl_id_34<v_N_0;v_gl_id_34 = (v_gl_id_34 + get_global_size(0))){
    float v_tmp_78 = 0.0f;
    v__47 = v_tmp_78;
    /* reduce_seq */
    for (int v_i_35 = 0;v_i_35<v_M_1;v_i_35 = (1 + v_i_35)){
      v__47 = multAndSumUp(v__47, v__41[v_i_35], v__40[(v_i_35 + (v_M_1 * v_gl_id_34))]);
    }
    /* end reduce_seq */
    /* map_seq */
    /* unroll */
    v__52_0 = id(v__47);
    /* end unroll */
    /* end map_seq */
    /* map_seq */
    /* unroll */
    v__54_0 = mult(v__43, v__52_0);
    /* end unroll */
    /* end map_seq */
    /* map_seq */
    /* unroll */
    v__56_0 = multAndSumUp(v__54_0, v__42[v_gl_id_34], v__44);
    /* end unroll */
    /* end map_seq */
    /* map_seq */
    /* unroll */
    v__57[v_gl_id_34] = id(v__56_0);
    /* end unroll */
    /* end map_seq */
  }
}}
