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
kernel void KERNEL(const global float* restrict v__48, const global float* restrict v__49, global float* v__55, int v_K_2, int v_M_1, int v_N_0){ 
#ifndef WORKGROUP_GUARD
#define WORKGROUP_GUARD
#endif
WORKGROUP_GUARD
{
  /* Static local memory */
  /* Typed Value memory */
  float v__51;
  /* Private Memory */
  for (int v_gl_id_44 = get_global_id(0);v_gl_id_44<v_M_1;v_gl_id_44 = (v_gl_id_44 + get_global_size(0))){
    for (int v_gl_id_45 = get_global_id(1);v_gl_id_45<v_N_0;v_gl_id_45 = (v_gl_id_45 + get_global_size(1))){
      float v_tmp_69 = 0.0f;
      v__51 = v_tmp_69;
      /* reduce_seq */
      for (int v_i_46 = 0;v_i_46<v_K_2;v_i_46 = (1 + v_i_46)){
        v__51 = multAndSumUp(v__51, v__48[(v_i_46 + (v_K_2 * v_gl_id_44))], v__49[(v_i_46 + (v_K_2 * v_gl_id_45))]);
      }
      /* end reduce_seq */
      /* map_seq */
      /* iteration count is exactly 1, no loop emitted */
      {
        int v_i_47 = 0;
        v__55[(v_gl_id_45 + (v_N_0 * v_gl_id_44))] = id(v__51);
      }
      /* end map_seq */
    }
  }
}}
