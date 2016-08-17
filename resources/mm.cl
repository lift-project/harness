float multAndSumUp(float acc, float l, float r){
  { return acc + (l * r); }
}
float id(float x){
  { return x; }
}
kernel void KERNEL(const global float* restrict v__14, const global float* restrict v__15, global float* v__22, int v_K_2, int v_M_1, int v_N_0){ 
#ifndef WORKGROUP_GUARD
#define WORKGROUP_GUARD
#endif
WORKGROUP_GUARD
{
  /* Static local memory */
  /* Typed Value memory */
  float v__17;
  /* Private Memory */
  for (int v_gl_id_10 = get_global_id(0);v_gl_id_10<v_M_1;v_gl_id_10 = (v_gl_id_10 + get_global_size(0))){
    for (int v_gl_id_11 = get_global_id(1);v_gl_id_11<v_N_0;v_gl_id_11 = (v_gl_id_11 + get_global_size(1))){
      float v_tmp_36 = 0.0f;
      v__17 = v_tmp_36;
      /* reduce_seq */
      for (int v_i_12 = 0;v_i_12<v_K_2;v_i_12 = (1 + v_i_12)){
        v__17 = multAndSumUp(v__17, v__14[(v_i_12 + (v_K_2 * v_gl_id_10))], v__15[(v_gl_id_11 + (v_N_0 * v_i_12))]);
      }
      /* end reduce_seq */
      /* map_seq */
      /* iteration count is exactly 1, no loop emitted */
      {
        int v_i_13 = 0;
        v__22[(v_gl_id_11 + (v_N_0 * v_gl_id_10))] = id(v__17);
      }
      /* end map_seq */
    }
  }
}}
