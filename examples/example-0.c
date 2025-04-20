#include "zor.h"
#include <stdio.h>
#include <stdlib.h>

#define PRINT(name, tensor)                                                    \
  do {                                                                         \
    char buf[256];                                                             \
    zor_to_string(tensor, buf, sizeof(buf));                                   \
    printf(#name " =\n%s\n", buf);                                             \
  } while (0)

int main() {
  zor_srandom(42);

  // 1. Basic tensors
  uint32_t shape[2] = {2, 3};
  void *Ones = zor_ones(2, shape);
  void *Threes = zor_fill(2, shape, 3.0f);
  void *Rand = zor_random(2, shape, -1.0f, 1.0f);

  PRINT(Ones, Ones);
  PRINT(Threes, Threes);
  PRINT(Rand, Rand);

  // 2. Arithmetic ops
  void *Sum = zor_add(Ones, Threes);
  void *Diff = zor_subtract(Threes, Ones);
  void *Prod = zor_multiply(Sum, Rand);
  void *Quot = zor_divide(Prod, Threes);

  PRINT(Sum, Sum);
  PRINT(Diff, Diff);
  PRINT(Prod, Prod);
  PRINT(Quot, Quot);

  // 3. Reductions
  void *Sum1 = zor_sum(Sum, 1);
  void *Mean0 = zor_mean(Prod, 0);
  void *Max1 = zor_max(Prod, 1);

  PRINT(Sum1, Sum1);
  PRINT(Mean0, Mean0);
  PRINT(Max1, Max1);

  // 4. Reshape and transpose
  uint32_t flat_shape[1] = {6};
  int32_t axes[2] = {1, 0};
  void *Flat = zor_reshape(Sum, 1, flat_shape);
  void *Trans = zor_transpose(Sum, axes);

  PRINT(Flat, Flat);
  PRINT(Trans, Trans);

  // 5. Slicing
  int32_t sl0[3] = {0, 1, 1};
  int32_t sl1[3] = {1, 3, 1};
  int32_t *slices[2] = {sl0, sl1};
  void *Slice = zor_slice(Sum, 2, slices);

  PRINT(Slice, Slice);

  // 6. Get/set element
  int idx[2] = {1, 2};
  float val;
  if (zor_get_element(Sum, idx, &val))
    printf("Sum[1][2] = %f\n", val);

  zor_set_element(Sum, idx, 9.99f);
  PRINT(Sum_modified, Sum);

  // 7. Matmul
  uint32_t A_shape[2] = {2, 3};
  uint32_t B_shape[2] = {3, 2};
  void *A = zor_random(2, A_shape, 0, 1);
  void *B = zor_random(2, B_shape, 0, 1);
  void *C = zor_matmul(A, B);

  PRINT(A, A);
  PRINT(B, B);
  PRINT(C, C);

  // 8. Tensordot
  int32_t a_axes[1] = {1};
  int32_t b_axes[1] = {0};
  void *TD = zor_tensordot(A, B, 1, a_axes, b_axes);

  PRINT(TD, TD);

  // 9. Elementwise nonlinearities
  void *Neg = zor_negative(Prod);
  void *ReLU = zor_relu(Prod);
  void *Sigmoid = zor_sigmoid(Prod);

  PRINT(Neg, Neg);
  PRINT(ReLU, ReLU);
  PRINT(Sigmoid, Sigmoid);

  // 10. Embedding lookup
  uint32_t tab_shape[2] = {10, 4};
  void *Table = zor_random(2, tab_shape, -1, 1);
  uint32_t toks_shape[1] = {3};
  float toks_data[3] = {2, 5, 8};
  void *Toks = zor_from_array(1, toks_shape, toks_data);
  void *Emb = zor_embed(Table, Toks);

  PRINT(Toks, Toks);
  PRINT(Emb, Emb);

  // 11. Softmax + backward
  void *Soft = zor_softmax(Prod, 1);
  void *Grad = zor_softmax_backward(Prod, Soft, 1);

  PRINT(Softmax, Soft);
  PRINT(Softmax_Grad, Grad);

  // 12. Size check
  uint64_t size = zor_size(Sum);
  printf("Size of Sum: %llu elements\n", (unsigned long long)size);

  // 13. Free all
  void *tensors[] = {Ones, Threes,  Rand,  Sum,   Diff, Prod, Quot, Sum1, Mean0,
                     Max1, Flat,    Trans, Slice, A,    B,    C,    TD,   Neg,
                     ReLU, Sigmoid, Table, Toks,  Emb,  Soft, Grad};
  for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); ++i)
    zor_free(tensors[i]);

  return 0;
}
