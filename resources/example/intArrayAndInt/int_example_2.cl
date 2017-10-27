kernel void KERNEL(global int* input, int add_this, global int* output, int N) {

  int gid = get_global_id(0);

  for (int i = gid; i < N; i += get_global_size(0)) {
    output[i] = input[i] + add_this;
  }
}
