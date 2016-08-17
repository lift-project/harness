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
kernel void KERNEL(const global float* restrict v__36, const global float* restrict v__37, global float* v__44, int v_K_2, int v_N_0){ 
#ifndef WORKGROUP_GUARD
#define WORKGROUP_GUARD
#endif
WORKGROUP_GUARD
{
  /* Static local memory */
  /* Typed Value memory */
  float v__39;
  /* Private Memory */
  for (int v_gl_id_33 = get_global_id(0);v_gl_id_33<v_N_0;v_gl_id_33 = (v_gl_id_33 + get_global_size(0))){
    float v_tmp_54 = 0.0f;
    v__39 = v_tmp_54;
    /* reduce_seq */
    for (int v_i_34 = 0;v_i_34<v_K_2;v_i_34 = (1 + v_i_34)){
      v__39 = multAndSumUp(v__39, v__36[(v_i_34 + (v_K_2 * v_gl_id_33))], v__37[v_i_34]);
    }
    /* end reduce_seq */
    /* map_seq */
    /* iteration count is exactly 1, no loop emitted */
    {
      int v_i_35 = 0;
      v__44[v_gl_id_33] = id(v__39);
    }
    /* end map_seq */
  }
}}
