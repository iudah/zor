#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define __STDC_WANT_IEC_60559_FUNC_EXT__ 1
#define __STDC_WANT_IEC_60559_DFB_EXT__ 1
#include "../include/zor.h"
#include "../include/zor_simd_defines.h"
#include <entropy.h>
#include <float.h>
#include <pcg_variants.h>
#include <zot.h>

typedef struct zor {
  zfl *data;
  uint32_t *shape;
  uint32_t *strides;
  uint64_t data_size;
  uint8_t rank;
} zor;

void *zor_init(uint8_t rank, uint32_t *restrict shape) {
  if (rank == 0 || shape == NULL) {
    LOG_ERROR(
        "Invalid input: Rank must be greater than 0, and shape cannot be NULL");
    return NULL;
  }

  zor *tensor = zcalloc(1, sizeof(*tensor));
  if (tensor == NULL) {
    LOG_ERROR("Memory allocation failed for tensor.");
    return NULL;
  }

  tensor->rank = rank;
  tensor->shape = zcalloc(rank, sizeof(*tensor->shape));
  if (tensor->shape == NULL) {
    LOG_ERROR("Memory allocation failed for tensor shape array.");
    zfree(tensor);
    return NULL;
  }

  tensor->strides = zcalloc(rank, sizeof(*tensor->shape));
  if (tensor->strides == NULL) {
    LOG_ERROR("Memory allocation failed for tensor stride array.");
    zfree(tensor->shape);
    zfree(tensor);
    return NULL;
  }

  tensor->data_size = 1;

  typeof(rank) i = rank;
  while (i) {
    i--;
    tensor->strides[i] = tensor->data_size;
    tensor->data_size *= tensor->shape[i] = shape[i];
  }

  tensor->data = zcalloc(tensor->data_size, sizeof(*tensor->data));
  if (tensor->data == NULL) {
    LOG_ERROR("Memory allocation failed for tensor data array.");
    zfree(tensor->strides);
    zfree(tensor->shape);
    zfree(tensor);
    return NULL;
  }

  return tensor;
}

uint8_t zor_rank(void *restrict tensor) { return ((zor *)tensor)->rank; }

uint32_t *zor_shape(void *restrict self, uint32_t *shape) {
  if (shape) {
    zor *restrict tensor = self;
    memcpy(shape, tensor->shape, tensor->rank * sizeof(*shape));
  }
  return shape;
}

void zor_swap_repr(void *restrict self, void *restrict other) {
  zor *restrict tensor = self;
  zor *restrict swap = other;

  uint32_t *restrict shape = tensor->shape;
  uint8_t rank = tensor->rank;
  auto strides = tensor->strides;
  auto data_size = tensor->data_size;
  auto data = tensor->data;

  tensor->shape = swap->shape;
  tensor->data = swap->data;
  tensor->shape = swap->shape;
  tensor->rank = swap->rank;
  tensor->data_size = swap->data_size;
  tensor->strides = swap->strides;

  swap->shape = shape;
  swap->data = data;
  swap->shape = shape;
  swap->rank = rank;
  swap->data_size = data_size;
  swap->strides = strides;
}

void zor_swap_array(void *restrict self, void *restrict other) {
  zor *restrict tensor = self;
  zor *restrict swap = other;

  auto data = tensor->data;

  tensor->data_size = swap->data_size;
  swap->data = data;
}

void zor_free(void *restrict self) {
  zor *restrict tensor = self;
  if (tensor == NULL)
    return; // Nothing to free
  if (tensor->data)
    zfree(tensor->data);
  if (tensor->strides)
    zfree(tensor->strides);
  if (tensor->shape)
    zfree(tensor->shape);
  zfree(tensor);
}

void *zor_zeros(uint8_t rank, uint32_t *restrict shape) {
  return zor_init(rank, shape);
}

static void *zor_test_linear(uint8_t rank, uint32_t *restrict shape) {
  zor *tensor = zor_init(rank, shape);
  if (tensor == NULL) {
    return NULL;
  }

  for (uint64_t i = 0; i < tensor->data_size; i++) {
    tensor->data[i] = (zfl)1. * i;
  }

  return tensor;
}

void *zor_ones(uint8_t rank, uint32_t *restrict shape) {
  return zor_fill(rank, shape, 1);
}

void *zor_fill(uint8_t rank, uint32_t *restrict shape, float value) {
  zor *tensor = zor_init(rank, shape);
  if (tensor == NULL) {
    return NULL;
  }

  for (uint64_t i = 0; i < tensor->data_size; i++) {
    tensor->data[i] = (zfl)value;
  }

  return tensor;
}

static bool pcg_is_seeded = false;

void zor_srandom(uint64_t seed) {
  if (!pcg_is_seeded)
    pcg_is_seeded = true;

  pcg32_srandom(seed, 54u);
}

void *zor_random(uint8_t rank, uint32_t *restrict shape, float min, float max) {
  if (max < min) {
    LOG_ERROR("Invalid range. Minimum value must be less than maximum value");
    return NULL;
  }

  zor *tensor = zor_init(rank, shape);
  if (tensor == NULL) {
    return NULL;
  }

  if (!pcg_is_seeded) {
    uint64_t seeds[2];
    entropy_getbytes((void *)seeds, sizeof(seeds));
    pcg32_srandom(seeds[0], seeds[1]);
    pcg_is_seeded = true;
  }

  zfl width = max - min;
  for (uint64_t i = 0; i < tensor->data_size; i++) {
    tensor->data[i] =
        (zfl)(min + width * (float)(pcg32_random() / (double)UINT32_MAX));
  }

  return tensor;
}

void *zor_from_array(uint8_t rank, uint32_t *shape, float *numbers) {

  zor *tensor = zor_init(rank, shape);
  if (tensor == NULL) {
    return NULL;
  }

  for (uint64_t i = 0; i < tensor->data_size; i++) {
    tensor->data[i] = (zfl)(numbers[i]);
  }

  return tensor;
}

void *zor_reshape(void *restrict self, uint8_t rank, uint32_t *shape) {
  zor *restrict tensor = self;
  if (tensor == NULL) {
    LOG_ERROR("Invalid input. Tensor must not be NULL");
    return NULL;
  }

  zor *reshaped_tensor = zor_init(rank, shape);
  if (reshaped_tensor == NULL) {
    return NULL;
  }

  if (reshaped_tensor->data_size != tensor->data_size) {
    zor_free(reshaped_tensor);
    LOG_ERROR("Data shape mismatch. Reshaped tensor size (%" PRIu64
              ") does not match original tensor size(%" PRIu64 ")",
              reshaped_tensor->data_size, tensor->data_size);
    return NULL;
  }

  for (uint64_t i = 0; i < tensor->data_size; i++) {
    reshaped_tensor->data[i] = tensor->data[i];
  }

  return reshaped_tensor;
}

void *zor_transpose(void *restrict self, int32_t *restrict axes) {
  zor *restrict tensor = self;

  if (tensor == NULL) {
    LOG_ERROR("Invalid input. Tensor must not be NULL");
    return NULL;
  }

  uint32_t t_shape[tensor->rank];
  uint32_t t_strides[tensor->rank];
  bool is_used_axis[tensor->rank];

  for (uint8_t i = 0; i < tensor->rank; i++) {
    is_used_axis[i] = false;
  }

  for (uint8_t i = 0; i < tensor->rank; i++) {
    if (axes[i] >= tensor->rank) {
      LOG_ERROR("Axis %" PRId32 " is out of bounds (rank: %" PRIu32 ")",
                axes[i], tensor->rank);
      return nullptr;
    }

    if (is_used_axis[axes[i]]) {
      LOG_ERROR("Duplicate axis %" PRId32 " detected. Each axis must be unique",
                axes[i]);
      return nullptr;
    }

    is_used_axis[axes[i]] = true;
    t_shape[i] = tensor->shape[axes[i]];
    t_strides[i] = tensor->strides[axes[i]];
  }

  zor *transpose = zor_init(tensor->rank, t_shape);
  if (transpose == NULL) {
    return NULL;
  }

  uint32_t indices[tensor->rank];
  for (uint8_t i = 0; i < tensor->rank; i++) {
    indices[i] = 0;
  }

  zsize offset;
  for (uint64_t i = 0; i < transpose->data_size; i++) {
    offset = 0;
    for (uint32_t i = 0; i < transpose->rank; i++) {
      offset += indices[i] * t_strides[i];
    }

    if (offset > tensor->data_size) {
      LOG_ERROR("Offset out of bounds. This is not supposed to happen. Please "
                "contact developer.");
      zor_free(transpose);
      return NULL;
    }

    transpose->data[i] = tensor->data[offset];

    for (uint8_t i = tensor->rank; i > 0;) {
      i--;
      indices[i]++;

      if (indices[i] >= transpose->shape[i]) {
        indices[i] = 0;
        continue;
      }

      break;
    }
  }

  return transpose;
}

void *zor_slice(void *self, uint32_t n_slice_triples,
                int32_t **restrict slice_triples) {
  zor *restrict tensor = self;

  if (tensor == NULL) {
    LOG_ERROR("Invalid input. Tensor must not be NULL");
    return NULL;
  }

  if (n_slice_triples == 0 || slice_triples == NULL) {
    LOG_ERROR("Invalid slicing parameters. Ensure slice_triples are valid and "
              "within bounds.");
  }

  uint32_t slice_shape[tensor->rank];
  uint32_t starts[tensor->rank];
  uint32_t steps[tensor->rank];
  uint8_t axis = 0, slice_index = 0;
  bool ellipsis_used = false;

  while (axis < tensor->rank && slice_index < n_slice_triples) {
    if (slice_triples[slice_index] == ELLIPSIS) {
      // Handle ellipsis
      if (ellipsis_used) {
        LOG_ERROR("Multiple ellipsis ('...') found in slicing parameters. Only "
                  "one ellipsis is allowed.");
        return NULL; // Only one ellipsis allowed
      }
      ellipsis_used = true;

      // Fill remaining axes with full slices
      int8_t remaining_axes = tensor->rank - n_slice_triples + 1;
      while (remaining_axes-- > 0) {
        slice_shape[axis + remaining_axes] =
            tensor->shape[axis + remaining_axes];
        starts[axis + remaining_axes] = 0;
        steps[axis + remaining_axes] = 1;
      }
      slice_index++;
      continue;
    }

    if (slice_triples[slice_index] == NULL) {
      // Slice all (":")
      slice_shape[axis] = tensor->shape[axis];
      starts[axis] = 0;
      steps[axis] = 1;
    } else {
      // Normal slice
      int *st = slice_triples[slice_index];
      int start = st[0], stop = st[1], step = st[2];

      if (start < 0)
        start += tensor->shape[axis]; // Handle negative indexing
      if (stop < 0)
        stop += tensor->shape[axis]; // Handle negative indexing

      starts[axis] = start;
      steps[axis] = step;

      if (start < 0 || start > stop || stop > tensor->shape[axis] ||
          step == 0) {
        if (start < 0)
          LOG_ERROR("Invalid start index %d for axis %d. Start must greater or "
                    "equal to 0 but less than stop.",
                    start, axis);
        if (stop > tensor->shape[axis])
          LOG_ERROR("Invalid stop index %d for axis %d. Stop must be less than "
                    "shape (%" PRIu32 ").",
                    stop, axis, tensor->shape[axis]);
        if (step == 0)
          LOG_ERROR("Invalid step %d for axis %d. Step must be non-zero.", step,
                    axis);
        return NULL;
      }
      slice_shape[axis] = (stop - start + step - 1) / step;
    }

    axis++;
    slice_index++;
  }

  // Initialize new tensor with computed slice shape
  zor *slice = zor_init(tensor->rank, slice_shape);
  if (!slice)
    return NULL;

  uint32_t indices[tensor->rank];
  for (uint8_t i = 0; i < tensor->rank; i++) {
    indices[i] = 0;
  }

  // Populate `slice->data` based on slice_triples
  zsize offset;
  for (uint64_t i = 0; i < slice->data_size; i++) {
    offset = 0;
    for (uint32_t i = 0; i < slice->rank; i++) {
      offset += (starts[i] + indices[i] * steps[i]) * tensor->strides[i];
    }

    slice->data[i] = tensor->data[offset];

    for (uint8_t i = tensor->rank; i > 0;) {
      i--;
      indices[i]++;

      if (indices[i] >= slice->shape[i]) {
        indices[i] = 0;
        continue;
      }

      break;
    }
  }

  return slice;
}

__attribute__((
    warn_unused_result("Ensure to check return value of function"))) bool
zor_get_element(void *restrict self, const int *restrict indices,
                float *restrict value) {
  zor *restrict tensor = self;

  if (tensor == NULL) {
    LOG_ERROR("Invalid input. Tensor must not be NULL");
    return NULL;
  }

  zsize offset = 0;
  for (uint32_t i = 0; i < tensor->rank; i++) {
    if (indices[i] >= tensor->shape[i]) {
      LOG_ERROR("Index %" PRIu32 " out of bounds for axis %d (shape: %" PRIu32
                "). Check indices.",
                i, indices[i], tensor->shape[i]);
      return false;
    }
    offset += indices[i] * tensor->strides[i];
  }
  *value = tensor->data[offset];
  return true;
}

bool zor_set_element(void *restrict self, const int *restrict indices,
                     float value) {
  zor *restrict tensor = self;

  if (tensor == NULL) {
    LOG_ERROR("Invalid input. Tensor must not be NULL");
    return NULL;
  }

  zsize offset = 0;
  for (uint32_t i = 0; i < tensor->rank; i++) {
    if (indices[i] >= tensor->shape[i]) {
      LOG_ERROR("Index %" PRIu32 " out of bounds for axis %d (shape: %" PRIu32
                "). Check indices.",
                i, indices[i], tensor->shape[i]);
      return false;
    }
    offset += indices[i] * tensor->strides[i];
  }

  tensor->data[offset] = (zfl)value;
  return true;
}

// Core math functionalities and utilities
#if defined(use_bfp16) || defined(use_float16)
#include <arm_neon.h>

#define SIMD_STRIDE 8
#if defined(use_bfp16)
#define LOAD_SIMD vld1q_bf16
#define DUP_N_SIMD vdupq_n_bf16
#define STORE_SIMD vst1q_bf16

#define SIMD_add vaddq_bf16
#define SIMD_subtract vsubq_bf16
#define SIMD_additive_inverse vnegq_bf16

#elif defined(use_float16)
#define LOAD_SIMD vld1q_f16
#define DUP_N_SIMD vdupq_n_f16
#define STORE_SIMD vst1q_f16
#define SIMD_initial_reciprocal vrecpeq_f16
#define SIMD_correction_factor vrecpsq_f16
#define SIMD_type float16x4_t

#define SIMD_add vaddq_f16
#define SIMD_sum vaddvq_f16
#define SIMD_subtract vsubq_f16
#define SIMD_multiply vmulq_f16
#define SIMD_divide vdivq_f16
#endif

#elif defined(use_simd_float32)
#include <arm_neon.h>
// #include <neon2sse.h>

#define SIMD_STRIDE 4
#define LOAD_SIMD vld1q_f32
#define DUP_N_SIMD vdupq_n_f32
#define STORE_SIMD vst1q_f32

#define SIMD_add vaddq_f32
#define SIMD_subtract vsubq_f32
#define SIMD_additive_inverse vnegq_f32

#else
#define SIMD_STRIDE 1
#endif

#if !(__ARM_FP & 2)
/*No hardware floating point support */

#if defined(use_simd_float32) || defined(use_bfp16)
#define SIMD_type float32x4_t
#define SIMD_initial_reciprocal vrecpeq_f32
#define SIMD_correction_factor vrecpsq_f32
#define SIMD_get_high vget_high_f32
#define SIMD_get_low vget_low_f32
#define SIMD_get_lane vget_lane_f32

#define SIMD_divide vdivq_f32
#define SIMD_multiply vmulq_f32

#define SIMD_add_x2 vadd_f32
#define SIMD_padd_x2 vpadd_f32
#define SIMD_sum vaddvq_f32

#define SIMD_min vminq_f32
#define SIMD_min_x2 vmin_f32
#define SIMD_pmin_x2 vpmin_f32
#define SIMD_reduce_min vminvq_f32

#define SIMD_max vmaxq_f32
#define SIMD_max_x2 vmax_f32
#define SIMD_pmax_x2 vpmax_f32
#define SIMD_reduce_max vmaxvq_f32

#endif

#define __ai static __inline__ __attribute__((__always_inline__, __nodebug__))

__ai __attribute__((target("neon"))) SIMD_type SIMD_divide(SIMD_type dividend,
                                                           SIMD_type divisor) {

  /*determine an initial estimate of reciprocal of divisor.*/
  auto initial_reciprocal = SIMD_initial_reciprocal(divisor);
  auto correction_factor = SIMD_correction_factor(divisor, initial_reciprocal);
  initial_reciprocal = SIMD_multiply(initial_reciprocal, correction_factor);
  correction_factor = SIMD_correction_factor(divisor, initial_reciprocal);
  initial_reciprocal = SIMD_multiply(initial_reciprocal, correction_factor);

  return SIMD_multiply(dividend, initial_reciprocal);
}

#if defined(use_float16)
#define float_type zfl
#else
#define float_type float
#endif

__ai __attribute__((target("neon"))) float_type SIMD_sum(SIMD_type a) {
  auto sum = SIMD_add_x2(SIMD_get_high(a), SIMD_get_high(a));
  sum = SIMD_padd_x2(sum, sum);

  return SIMD_get_lane(sum, 0);
}

__ai __attribute__((target("neon"))) float_type SIMD_reduce_min(SIMD_type a) {
  auto min = SIMD_min_x2(SIMD_get_high(a), SIMD_get_high(a));
  min = SIMD_pmin_x2(min, min);
  return SIMD_get_lane(min, 0);
}

__ai __attribute__((target("neon"))) float_type SIMD_reduce_max(SIMD_type a) {
  auto max = SIMD_max_x2(SIMD_get_high(a), SIMD_get_high(a));
  max = SIMD_pmin_x2(max, max);
  return SIMD_get_lane(max, 0);
}

#if defined(use_bfp16)
#undef SIMD_divide
#define SIMD_divide vdivq_bf16

#undef SIMD_multiply
#define SIMD_multiply vmulq_bf16

#undef SIMD_type
#define SIMD_type bfloat16x8_t

#undef SIMD_sum
#define SIMD_sum vaddvq_bf16

#undef SIMD_min
#define SIMD_min vminq_bf16

#undef SIMD_max
#define SIMD_max vmaxq_bf16

#undef SIMD_reduce_min
#define SIMD_reduce_min vminvq_bf16

#undef SIMD_reduce_max
#define SIMD_reduce_max vmaxvq_bf16

__ai __attribute__((target("neon"))) SIMD_type simd_operate(
    SIMD_type a, SIMD_type b, float32x4_t(operator)(float32x4_t, float32x4_t)) {
  auto a_low = vcvtq_low_f32_bf16(a);
  auto a_high = vcvtq_high_f32_bf16(a);

  auto b_low = vcvtq_low_f32_bf16(b);
  auto b_high = vcvtq_high_f32_bf16(b);

  auto result_low = operator(a_low, b_low);
  auto result_high = operator(a_high, b_high);

  return vcombine_bf16(vcvt_bf16_f32(result_low), vcvt_bf16_f32(result_high));
}

__ai __attribute__((target("neon"))) SIMD_type SIMD_add(SIMD_type a,
                                                        SIMD_type b) {
  return simd_operate(a, b, vaddq_f32);
}
__ai __attribute__((target("neon"))) SIMD_type SIMD_subtract(SIMD_type a,
                                                             SIMD_type b) {
  return simd_operate(a, b, vsubq_f32);
}
__ai __attribute__((target("neon"))) SIMD_type SIMD_multiply(SIMD_type a,
                                                             SIMD_type b) {
  return simd_operate(a, b, vmulq_f32);
}
__ai __attribute__((target("neon"))) SIMD_type SIMD_divide(SIMD_type a,
                                                           SIMD_type b) {
  return simd_operate(a, b, vdivq_f32);
}
__ai __attribute__((target("neon"))) SIMD_type SIMD_min(SIMD_type a,
                                                        SIMD_type b) {
  return simd_operate(a, b, vminq_f32);
}
__ai __attribute__((target("neon"))) SIMD_type SIMD_max(SIMD_type a,
                                                        SIMD_type b) {
  return simd_operate(a, b, vmaxq_f32);
}

__ai __attribute__((target("neon"))) zfl SIMD_sum(SIMD_type a) {
  auto a_low = vcvtq_low_f32_bf16(a);
  auto a_high = vcvtq_high_f32_bf16(a);

  return (zfl)(vaddvq_f32(a_low) + vaddvq_f32(a_high));
}
__ai __attribute__((target("neon"))) zfl SIMD_reduce_max(SIMD_type a) {
  auto a_low = vcvtq_low_f32_bf16(a);
  auto a_high = vcvtq_high_f32_bf16(a);

  return (zfl)fmaxf(vmaxvq_f32(a_low), vmaxvq_f32(a_high));
}
__ai __attribute__((target("neon"))) zfl SIMD_reduce_min(SIMD_type a) {
  auto a_low = vcvtq_low_f32_bf16(a);
  auto a_high = vcvtq_high_f32_bf16(a);

  return (zfl)fminf(vminvq_f32(a_low), vminvq_f32(a_high));
}

#endif
#endif

#define maxi(a, b) ((a) > (b) ? a : b)

static bool is_element_wise_compatible(zor *a, zor *b, bool *require_broadcast,
                                       uint32_t *shape) {
  auto a_rank = a->rank;
  auto b_rank = b->rank;

  while (a_rank && b_rank) {
    a_rank--;
    b_rank--;

    if (a->shape[a_rank] != b->shape[b_rank]) {
      if (a->shape[a_rank] != 1 && b->shape[b_rank] != 1) {
        return false;
      }
      *require_broadcast = true;
    }

    shape[maxi(a_rank, b_rank)] = maxi(a->shape[a_rank], b->shape[b_rank]);
  }

  if (a_rank || b_rank)
    *require_broadcast = true;

  while (a_rank) {
    a_rank--;
    shape[a_rank] = a->shape[a_rank];
  }
  while (b_rank) {
    b_rank--;
    shape[b_rank] = b->shape[b_rank];
  }

  return true;
}

static uint64_t compute_offset(zor *tensor, uint32_t *index) {
  uint64_t offset = 0;
  uint8_t i = 0;
  while (i < tensor->rank) {
    offset += tensor->strides[i] * index[i];
    i++;
  }
  return offset;
}

static void increment_pairwise_indices(zor *tensor, zor *a, zor *b,
                                       uint32_t *result_index,
                                       uint32_t *a_index, uint32_t *b_index) {
  auto a_rank_difference = tensor->rank - a->rank;
  auto b_rank_difference = tensor->rank - b->rank;
  for (auto i = tensor->rank; i > 0;) {
    i--;
    result_index[i]++;
    if (a->shape[i - a_rank_difference] > 1) {
      a_index[i - a_rank_difference]++;
      if (a_index[i - a_rank_difference] >= a->shape[i - a_rank_difference])
        a_index[i - a_rank_difference] = 0;
    }
    if (b->shape[i - b_rank_difference] > 1) {
      b_index[i - b_rank_difference]++;
      if (b_index[i - b_rank_difference] >= b->shape[i - b_rank_difference])
        b_index[i - b_rank_difference] = 0;
    }
    if (result_index[i] >= tensor->shape[i]) {
      result_index[i] = 0;
      continue;
    }
    break;
  }
}
void *zor_pairwise(void *self, void *other,
                   zfl (*scalar_binary_operation)(zfl a, zfl b),
                   SIMD_type (*simd_binary_operation)(SIMD_type a,
                                                      SIMD_type b)) {
  zor *a = self, *b = other;
  if (a == NULL || b == NULL) {
    LOG_ERROR("Invalid input. Tensor must not be NULL. Please ensure the "
              "tensor is properly initialized before using it.");
    return NULL;
  }
  uint32_t shape[maxi(a->rank, b->rank)];
  bool require_broadcast = false;
  if (!is_element_wise_compatible(a, b, &require_broadcast, shape)) {
    LOG_ERROR("Incompatible shapes for addition. Please ensure "
              "the shapes ");
    return NULL;
  }
  zor *result = zor_init(maxi(a->rank, b->rank), shape);
  if (!result)
    return NULL;
  uint64_t total_rank;
  uint32_t indices[total_rank = a->rank + b->rank + result->rank];
  auto a_index = indices;
  auto b_index = a_index + a->rank;
  auto result_index = b_index + b->rank;
  if (require_broadcast) {
    memset(indices, 0, sizeof(indices));
  }
  uint32_t a_i = 0, b_i = 0;
  zfl bc[SIMD_STRIDE * 2];
  zfl *a_data = bc;
  zfl *b_data = bc + SIMD_STRIDE;
  uint32_t a_offset = 0, b_offset = 0;
  const auto a_rank_difference = result->rank - a->rank;
  const auto b_rank_difference = result->rank - b->rank;
  uint64_t i = 0;
  for (; (i + SIMD_STRIDE) <= result->data_size; i += SIMD_STRIDE) {
    if (require_broadcast) {
      for (auto i = 0; i < SIMD_STRIDE; i++) {
        a_offset = compute_offset(a, a_index);
        b_offset = compute_offset(b, b_index);
        a_data[i] = a->data[a_offset];
        b_data[i] = b->data[b_offset];
        increment_pairwise_indices(result, a, b, result_index, a_index,
                                   b_index);
      }
    } else {
      a_data = a->data + i;
      b_data = b->data + i;
      a_offset = b_offset = i;
    }
#if SIMD_STRIDE > 1
    if (simd_binary_operation) {
      auto a_simd_vector = LOAD_SIMD(((a_data)));
      auto b_simd_vector = LOAD_SIMD(((b_data)));
      auto simd_result = simd_binary_operation(a_simd_vector, b_simd_vector);

      STORE_SIMD((result->data + i), simd_result);
    } else {
#endif
      if (scalar_binary_operation) {
        result->data[i] =
            (zfl)scalar_binary_operation(a->data[a_offset], b->data[b_offset]);
      }
#if SIMD_STRIDE > 1
      else {
        LOG_ERROR("Neither SIMD function nor scalar funtion has been provided. "
                  "Please check input function.");
      }
    }
#endif
  }
  while (i < result->data_size) {
    if (require_broadcast) {
      a_offset = compute_offset(a, a_index);
      b_offset = compute_offset(b, b_index);
    } else {
      a_offset = b_offset = i;
    }
    if (scalar_binary_operation) {
      result->data[i] =
          (zfl)scalar_binary_operation(a->data[a_offset], b->data[b_offset]);
    }
    if (require_broadcast) {
      increment_pairwise_indices(result, a, b, result_index, a_index, b_index);
    }
    i++;
  }
  return result;
}

struct unary_pairwise_data {
  zfl scalar;
  bool scalar_is_first;
};

void *zor_unary_pairwise(void *self, struct unary_pairwise_data *scalar_float,
                         zfl (*scalar_binary_operation)(zfl a, zfl b),
                         SIMD_type (*simd_binary_operation)(SIMD_type a,
                                                            SIMD_type b)) {
  zor *a = self;
  struct unary_pairwise_data *other = scalar_float;
  if (a == NULL || other == NULL) {
    LOG_ERROR("Invalid input. Tensor must not be NULL. Please ensure the "
              "tensor is properly initialized before using it.");
    return NULL;
  }
  uint32_t *shape = a->shape;
  bool require_broadcast = false;
  // if (!is_element_wise_compatible(a, b, &require_broadcast, shape)) {
  //   LOG_ERROR("Incompatible shapes for addition. Please ensure "
  //             "the shapes ");
  //   return NULL;
  // }
  zor *result = zor_init(a->rank, shape);
  if (!result)
    return NULL;
  uint64_t total_rank;
  uint32_t indices[total_rank = a->rank + result->rank];
  auto a_index = indices;
  auto result_index = a_index + a->rank;
  if (require_broadcast) {
    memset(indices, 0, sizeof(indices));
  }
  uint32_t a_i = 0;
  zfl bc[SIMD_STRIDE];
  zfl *a_data = bc;
  uint32_t a_offset = 0;
  const auto a_rank_difference = result->rank - a->rank;

  auto other_simd_vector = DUP_N_SIMD(((other->scalar)));
  uint64_t i = 0;
  for (; (i + SIMD_STRIDE) <= result->data_size; i += SIMD_STRIDE) {
    if (require_broadcast) {
    } else {
      a_data = a->data + i;
      a_offset = i;
    }
#if SIMD_STRIDE > 1
    if (simd_binary_operation) {
      auto a_simd_vector = LOAD_SIMD(((a_data)));
      auto simd_result = simd_binary_operation(
          other->scalar_is_first ? other_simd_vector : a_simd_vector,
          other->scalar_is_first ? a_simd_vector : other_simd_vector);

      STORE_SIMD((result->data + i), simd_result);
    } else {
#endif
      if (scalar_binary_operation) {
        result->data[i] = (zfl)scalar_binary_operation(

            other->scalar_is_first ? other->scalar : a->data[a_offset],
            other->scalar_is_first ? a->data[a_offset] : other->scalar);
      }
#if SIMD_STRIDE > 1
      else {
        LOG_ERROR("Neither SIMD function nor scalar funtion has been provided. "
                  "Please check input function.");
      }
    }
#endif
  }
  while (i < result->data_size) {
    if (require_broadcast) {
    } else {
      a_offset = i;
    }
    if (scalar_binary_operation) {
      result->data[i] = (zfl)scalar_binary_operation(
          other->scalar_is_first ? other->scalar : a->data[a_offset],
          other->scalar_is_first ? a->data[a_offset] : other->scalar);
    }
    i++;
  }
  return result;
}

static zfl scalar_add(zfl a, zfl b) { return a + b; }
void *zor_add(void *a, void *b) {
  return zor_pairwise(a, b, scalar_add, SIMD_add);
}

static zfl scalar_subtract(zfl a, zfl b) { return a - b; }
void *zor_subtract(void *a, void *b) {
  return zor_pairwise(a, b, scalar_subtract, SIMD_subtract);
}

static zfl scalar_multiply(zfl a, zfl b) { return a * b; }
void *zor_multiply(void *a, void *b) {
  return zor_pairwise(a, b, scalar_multiply, SIMD_multiply);
}

static zfl scalar_divide(zfl a, zfl b) { return a / b; }
void *zor_divide(void *a, void *b) {
  return zor_pairwise(a, b, scalar_divide, SIMD_divide);
}

static void perform_global_reduction(
    zor *restrict tensor, zor *restrict reduce,
    zfl (*scalar_binary_operation)(zfl a, zfl b),
    SIMD_type (*simd_binary_operation)(SIMD_type a, SIMD_type b),
    zfl (*simd_reduction_operation)(SIMD_type a)) {

  uint64_t i = 0;
#if SIMD_STRIDE > 1
  if (simd_binary_operation && tensor->data_size >= SIMD_STRIDE) {
    auto reduction_vector = LOAD_SIMD(tensor->data);
    typeof(reduction_vector) next_vector;
    for (i = SIMD_STRIDE; (i + SIMD_STRIDE) <= tensor->data_size;
         i += SIMD_STRIDE) {

      next_vector = LOAD_SIMD((tensor->data + i));
      reduction_vector = simd_binary_operation(reduction_vector, next_vector);
    }

    if (simd_reduction_operation) {
      reduce->data[0] = simd_reduction_operation(reduction_vector);
    } else {
      zfl reduction[SIMD_STRIDE];
      STORE_SIMD(reduction, reduction_vector);

      auto j = 0;
      while (j < SIMD_STRIDE) {
        reduce->data[0] =
            scalar_binary_operation(reduce->data[0], tensor->data[j++]);
      }
    }
  }
#endif
  if (scalar_binary_operation) {
    while (tensor->data_size - i) {
      reduce->data[0] =
          scalar_binary_operation(reduce->data[0], tensor->data[i++]);
    }
  }
}

static void perform_axis_reduction(
    zor *restrict tensor, zor *restrict reduce, uint8_t axis,
    zfl (*scalar_binary_operation)(zfl a, zfl b),
    SIMD_type (*simd_binary_operation)(SIMD_type a, SIMD_type b)) {

  auto axis_stride = tensor->strides[axis];
  auto reduced_size = reduce->data_size;
  for (auto i = 0; i < reduced_size;) {
    auto offset = (i / axis_stride) * (axis_stride * tensor->shape[axis]) +
                  (i % axis_stride);
    zfl reduced_value;
    auto j = 1;

#if SIMD_STRIDE > 1
    if (simd_binary_operation && tensor->strides[axis] >= SIMD_STRIDE &&
        (reduced_size - i) > SIMD_STRIDE) {
      auto reduced_vector = LOAD_SIMD(tensor->data + offset);
      for (; j < tensor->shape[axis]; j++) {
        auto vector = LOAD_SIMD(tensor->data + (offset + j * axis_stride));
        reduced_vector = simd_binary_operation(reduced_vector, vector);
      }

      STORE_SIMD(reduce->data + offset, reduced_vector);
      i += SIMD_STRIDE;
    } else {
#endif
      if (scalar_binary_operation) {
        reduced_value = tensor->data[offset];
        for (; j < tensor->shape[axis]; j++) {
          auto value = tensor->data[offset + j * axis_stride];
          reduced_value = scalar_binary_operation(reduced_value, value);
        }
        reduce->data[offset] = reduced_value;
        i += 1;
      }
#if SIMD_STRIDE > 1
    }
#endif
  }
}

void *zor_reduce(void *restrict self, int32_t reduce_axis, int32_t *axis_ptr,
                 zfl (*scalar_binary_operation)(zfl a, zfl b),
                 SIMD_type (*simd_binary_operation)(SIMD_type a, SIMD_type b),
                 zfl (*simd_reduction_operation)(SIMD_type)) {
  zor *tensor = self;
  if (tensor == NULL) {
    LOG_ERROR("Invalid input. Tensor must not be NULL");
    return NULL;
  }

  if (reduce_axis < 0)
    *axis_ptr = reduce_axis += tensor->rank;
  if (reduce_axis > tensor->rank && reduce_axis != ZOR_REDUCE_AXIS_NONE) {
    LOG_ERROR("Invalid reduction axis %" PRId32 ". Expected a value between 0 "
              "and %" PRIu32 ", or `ZOR_REDUCE_AXIS_NONE`.",
              reduce_axis, tensor->rank);
    return NULL;
  }

  if (!(scalar_binary_operation || simd_binary_operation ||
        simd_reduction_operation)) {
    LOG_ERROR("Neither SIMD function nor scalar funtion has been provided. "
              "Please check input function.");
    return NULL;
  }

  uint32_t output_shape[tensor->rank];

  if (reduce_axis == ZOR_REDUCE_AXIS_NONE) {
    // Set all dimensions to 1 for a scalar reduction
    for (uint32_t i = 0; i < tensor->rank; ++i) {
      output_shape[i] = 1;
    }
  } else {
    memcpy(output_shape, tensor->shape, sizeof(output_shape));
    // Collapse the reduced dimension
    output_shape[reduce_axis] = 1;
  }

  zor *result = zor_init(tensor->rank, output_shape);
  if (!result) {
    LOG_ERROR("Failed to initialize output tensor.");
    return NULL;
  }

  if (reduce_axis == ZOR_REDUCE_AXIS_NONE) {
    perform_global_reduction(tensor, result, scalar_binary_operation,
                             simd_binary_operation, simd_reduction_operation);
  } else {
    auto pre_reduce_transpose = tensor;
    if (reduce_axis) {
      int32_t reduce_transpose[tensor->rank];
      reduce_transpose[0] = reduce_axis;
      auto k = 1;
      for (auto i = 0; i < tensor->rank; i++) {
        if (i != reduce_axis)
          reduce_transpose[k++] = i;
      }

      pre_reduce_transpose =
          zor_transpose(pre_reduce_transpose, reduce_transpose);
    }

    perform_axis_reduction(pre_reduce_transpose, result, 0,
                           scalar_binary_operation, simd_binary_operation);
    if (reduce_axis) {

      zor_free(pre_reduce_transpose);
    }
  }
  return result;
}

void *zor_sum(void *restrict tensor, int axis) {
  return zor_reduce(tensor, axis, NULL, scalar_add, SIMD_add, SIMD_sum);
}

void *zor_mean(void *restrict self, int axis) {
  zor *tensor = self;
  zor *mean = zor_reduce(tensor, axis, &axis, scalar_add, SIMD_add, SIMD_sum);
  if (mean) {

    zfl divisor = (zfl)(axis != ZOR_REDUCE_AXIS_NONE ? tensor->shape[axis]
                                                     : tensor->data_size);

    if (mean->data_size >= SIMD_STRIDE) {
#if SIMD_STRIDE > 1
      auto divisor_vector = DUP_N_SIMD(divisor);
#endif
      auto i = 0;
#if SIMD_STRIDE > 1
      for (; (i - SIMD_STRIDE) <= mean->data_size; i += SIMD_STRIDE) {
        auto dividend_vector = LOAD_SIMD(((mean->data + i)));
        auto simd_result = SIMD_divide(dividend_vector, divisor_vector);

        STORE_SIMD((mean->data + i), simd_result);
      }
#endif

      for (; i < mean->data_size; i++) {
        mean->data[i] /= divisor;
      }
    }
  }

  return mean;
}

static zfl scalar_min(zfl a, zfl b) { return fminf((float)a, (float)b); }
void *zor_min(void *restrict tensor, int axis) {
  return zor_reduce(tensor, axis, NULL, scalar_min, SIMD_min, SIMD_reduce_min);
}
static zfl scalar_max(zfl a, zfl b) { return fmaxf((float)a, (float)b); }
void *zor_max(void *restrict tensor, int axis) {
  return zor_reduce(tensor, axis, NULL, scalar_max, SIMD_max, SIMD_reduce_max);
}

void *zor_matmul(void *self, void *other) {
  zor *a = self, *b = other;
  if (!a || !b) {
    LOG_ERROR("Input tensors cannot be NULL.");
    return NULL;
  }
  if (a->rank < 2 || b->rank < 2) {
    LOG_ERROR("Matrix multiplication requires tensors with rank >= 2. Received "
              "A.rank = %" PRIu8 ", B.rank = %" PRIu8 ".",
              a->rank, b->rank);
    return NULL;
  }
  if (a->shape[a->rank - 1] != b->shape[b->rank - 2]) {
    LOG_ERROR("Shapes are incompatible for matrix multiplication. A.shape[%d] "
              "= %" PRIu32 " does not match B.shape[%d] = %" PRIu32 ".",
              a->rank - 1, a->shape[a->rank - 1], b->rank - 2,
              b->shape[b->rank - 2]);
    return NULL;
  }

  int res_rank = a->rank + b->rank - 2;
  zor *transposed_b = b;

  if (b->rank > 2) {
    int transpose[b->rank];
    transpose[0] = b->rank - 2;
    int k = 1;
    for (int i = 0; i < b->rank; i++) {
      if (i != transpose[0])
        transpose[k++] = i;
    }
    transposed_b = zor_transpose(b, transpose);
    if (!transposed_b) {
      LOG_ERROR("Failed to transpose tensor b.");
      return NULL;
    }
  }

  uint32_t shape[res_rank];
  memcpy(shape, a->shape, sizeof(int) * (a->rank - 1));
  memcpy(shape + a->rank - 1, transposed_b->shape + 1,
         sizeof(int) * (transposed_b->rank - 1));

  zor *res = zor_init(res_rank, shape);
  if (!res) {
    LOG_ERROR("Failed to allocate result tensor.");
    if (b != transposed_b)
      zor_free(transposed_b);
    return NULL;
  }

  int common_dims = transposed_b->shape[0];
  int rows = a->data_size / common_dims;
  int cols = transposed_b->data_size / common_dims;

  int CACHE_LEN = 16; // Adjust based on hardware

  for (int j = 0; j < common_dims; j += CACHE_LEN) {
    for (int c = 0; c < cols; c += CACHE_LEN) {
      for (int r = 0; r < rows; r++) {
        for (int m = 0; m < CACHE_LEN && (m + j < common_dims); m++) {
          for (int k = 0; k < CACHE_LEN && (c + k < cols); k++) {
            res->data[r * cols + c + k] +=
                a->data[r * common_dims + j + m] *
                transposed_b->data[(j + m) * cols + c + k];
          }
        }
      }
    }
  }

  if (b != transposed_b)
    zor_free(transposed_b);
  return res;
}

void *zor_tensordot(void *self, void *other, int32_t n_axes, int32_t *a_axes,
                    int32_t *b_axes) {
  zor *a = self, *b = other;
  if (!a || !b) {
    LOG_ERROR("Input tensors cannot be NULL.");
    return NULL;
  }

  if (n_axes == 0) {
    LOG_ERROR("Number of contracting axes must be greater than 0.");
    return NULL;
  }

  if (a->rank == 0 || b->rank == 0)
    return zor_multiply(a, b);

  bool a_axes_used[a->rank];
  bool b_axes_used[b->rank];
  auto i = a->rank, j = b->rank;
  while (i || j) {
    if (i)
      a_axes_used[--i] = false;
    if (j)
      b_axes_used[--j] = false;
  }

  auto a_untouched_length = 1;
  auto b_untouched_length = 1;
  auto common_touched_length = 1;

  auto n_a_untouched_axes = a->rank - n_axes;
  auto n_b_untouched_axes = b->rank - n_axes;

  bool transpose_a = false;

  int32_t a_transpose[a->rank];
  int32_t b_transpose[b->rank];

  for (auto i = 0; i < n_axes; i++) {
    if (a_axes[i] < 0) {
      a_axes[i] += a->rank;
    }
    if (b_axes[i] < 0) {
      b_axes[i] += b->rank;
    }
    if (a_axes[i] < 0 || b_axes[i] < 0 || a_axes[i] >= a->rank ||
        b_axes[i] >= b->rank) {
      LOG_ERROR("Axes out of bounds.");
      return NULL;
    }
    if (a_axes_used[a_axes[i]] || b_axes_used[b_axes[i]]) {
      LOG_ERROR("Axes already used.");
      return NULL;
    }
    if (a->shape[a_axes[i]] != b->shape[b_axes[i]]) {
      LOG_ERROR("Axes do not match.");
      return NULL;
    }
    if (a_axes[i] != i)
      transpose_a = true;

    (a_transpose + n_a_untouched_axes)[i] = a_axes[i];
    b_transpose[i] = b_axes[i];

    common_touched_length *= b->shape[b_axes[i]];

    a_axes_used[a_axes[i]] = true;
    b_axes_used[b_axes[i]] = true;
  }

  uint32_t result_rank = a->rank - n_axes + b->rank - n_axes;
  uint32_t *result_shape = zcalloc(result_rank, sizeof(*result_shape));
  if (!result_shape) {
    LOG_ERROR("Failed to allocate result shape.");
    return NULL;
  }

  i = j = 0;
  auto m = 0, n = n_b_untouched_axes, k = n_a_untouched_axes;
  while (n < b->rank || (m < n_a_untouched_axes)) {
    if ((i < a->rank)) {
      if (!a_axes_used[i]) {
        a_transpose[m] = i;
        a_untouched_length *= result_shape[m] = a->shape[i];
        m++;
      }
    }
    if (i < b->rank) {
      if (!b_axes_used[i]) {
        b_transpose[n++] = i;
        b_untouched_length *= result_shape[k++] = b->shape[i];
      }
    }
    i++;
  }

  zor *transposed_a = transpose_a ? zor_transpose(a, a_transpose) : a;
  zor *transposed_b = zor_transpose(b, b_transpose);

  uint32_t a_shape[] = {a_untouched_length, common_touched_length, 1};
  uint32_t b_shape[] = {common_touched_length, b_untouched_length, 1};
  auto a_original_shape = a->shape;
  auto a_original_stride = a->strides;

  void *transpose_cache[] = {
      transposed_a->shape,
      transposed_a->strides,
      transposed_b->shape,
      transposed_b->strides,

  };

  uint8_t rank_cache[] = {
      transposed_a->rank,
      transposed_b->rank,
  };

  transposed_a->rank = 2;
  transposed_b->rank = 2;
  transposed_a->shape = a_shape;
  transposed_a->strides = a_shape + 1;
  transposed_b->shape = b_shape;
  transposed_b->strides = b_shape + 1;

  zor *result = zor_matmul(transposed_a, transposed_b);
  zfree(result->shape);
  zfree(result->strides);

  result->shape = result_shape;
  result->rank = result_rank;
  result->strides = zcalloc(result->rank, sizeof(*result->strides));
  result->strides[0] = 1;
  for (int i = 1; i < result->rank; i++) {
    result->strides[i] = result->strides[i - 1] * result->shape[i];
  }

  transposed_a->rank = rank_cache[0];
  transposed_b->rank = rank_cache[1];
  transposed_a->shape = transpose_cache[0];
  transposed_a->strides = transpose_cache[1];
  transposed_b->shape = transpose_cache[2];
  transposed_b->strides = transpose_cache[3];

  if (transposed_a == a) {
    a->rank = n_axes + n_a_untouched_axes;
    a->shape = a_original_shape;
    a->strides = a_original_stride;
  } else {
    zor_free(transposed_a);
  }
  zor_free(transposed_b);

  return result;
}

void *zor_copy(void *restrict self) {
  zor *restrict tensor = self;

  if (!tensor) {
    LOG_ERROR("Input tensors cannot be NULL.");
    return NULL;
  }

  zor *result = zor_init(tensor->rank, tensor->shape);
  if (!result) {
    LOG_ERROR("Failed to initialize output tensor.");
    return NULL;
  }

  uint64_t i = 0;
#if SIMD_STRIDE > 1
  for (i = 0; (i + SIMD_STRIDE) < tensor->data_size; i += SIMD_STRIDE) {
    auto simd_copy = LOAD_SIMD(tensor->data + i);
    STORE_SIMD(result->data + i, simd_copy);
  }
#endif
  for (; i < tensor->data_size; i++) {
    result->data[i] = tensor->data[i];
  }
  return result;
}

void *zor_negative(void *restrict self) {
  zor *restrict tensor = self;

  if (!tensor) {
    LOG_ERROR("Input tensors cannot be NULL.");
    return NULL;
  }

  zor *result = zor_init(tensor->rank, tensor->shape);
  if (!result) {
    LOG_ERROR("Failed to initialize output tensor.");
    return NULL;
  }

  uint64_t i = 0;
#if SIMD_STRIDE > 1
  // auto neg = DUP_N_SIMD(-1.);
  for (i = 0; (i + SIMD_STRIDE) < tensor->data_size; i += SIMD_STRIDE) {
    auto vec = LOAD_SIMD(tensor->data + i);
    auto neg_vec = SIMD_additive_inverse(vec);
    STORE_SIMD(result->data + i, neg_vec);
  }
#endif
  for (; i < tensor->data_size; i++) {
    result->data[i] = -tensor->data[i];
  }
  return result;
}

void *zor_relu(void *restrict self) {
  zor *restrict tensor = self;

  if (!tensor) {
    LOG_ERROR("Input tensors cannot be NULL.");
    return NULL;
  }

  volatile zor *result = zor_init(tensor->rank, tensor->shape);
  if (!result) {
    LOG_ERROR("Failed to initialize output tensor.");
    return NULL;
  }

  uint64_t i = 0;
#if SIMD_STRIDE > 1
  auto zero = DUP_N_SIMD(0.);
  for (i = 0; (i + SIMD_STRIDE) < tensor->data_size; i += SIMD_STRIDE) {
    auto vec = LOAD_SIMD(tensor->data + i);
    auto relu_vec = SIMD_max(zero, vec);

    STORE_SIMD(result->data + i, relu_vec);
  }
#endif
  for (; i < tensor->data_size; i++) {
    result->data[i] = fmax((double)tensor->data[i], 0);
  }
  return result;
}

uint64_t zor_to_string(void *tensor, char *buffer, uint64_t buffer_limit) {

  const zor *self = tensor;

  uint64_t col = self->shape[self->rank - 1];
  const zfl *const data = self->data;
  uint64_t data_length = self->data_size;
  uint64_t length = data_length <= 100 ? data_length : 100;
  uint64_t rear = 0, back = 0;

  // Use a temporary buffer to avoid overwriting the main buffer
  char tmp_buf[64];

  // First pass to determine rear and back with fixed decimal format
  for (uint64_t i = 0; i < length; i++) {
    int written = snprintf(tmp_buf, sizeof(tmp_buf), "%.15f", data[i]);
    if (written < 0) {
      return 0;
    }

    char *integer_part = tmp_buf;
    char *dot = strchr(tmp_buf, '.');
    uint64_t rtmp, btmp = 0;

    if (dot) {
      *dot = '\0';
      // Trim trailing zeros from the decimal part
      char *decimal_part = dot + 1;
      size_t decimal_len = strlen(decimal_part);
      while (decimal_len > 0 && decimal_part[decimal_len - 1] == '0') {
        decimal_len--;
      }
      btmp = decimal_len;
    }

    rtmp = strlen(integer_part);
    rear = rear > rtmp ? rear : rtmp;
    back = back > btmp ? back : btmp;
  }

  // Prepare the format string
  char fmt[255];
  snprintf(fmt, sizeof(fmt), "%%%u.%uf  ", (uint32_t)(rear + 1),
           (uint32_t)back);

  uint64_t i = 0, string_length = 0;
  while (i < length) {
    // Check remaining space to avoid overflow
    uint64_t remaining_space = buffer_limit - string_length;
    if (remaining_space <
        (rear + back + 2)) { // Estimated space for one element and newline
      break;
    }

    // Format the current number
    int written =
        snprintf(buffer + string_length, remaining_space, fmt, data[i]);
    if (written < 0) {
      break;
    }
    string_length += written;

    // Add newline at the end of each row
    if ((i % col) == (col - 1)) {
      if (string_length < buffer_limit) {
        buffer[string_length] = '\n';
        string_length++;
        buffer[string_length] = '\0';
      } else {
        break;
      }
    }

    i++;
  }

  // Add truncation indicator if necessary
  if (data_length > i) {
    // Ensure at least 4 bytes available for "...\0"
    uint64_t available = buffer_limit - string_length;
    if (available >= 4) {
      strcpy(buffer + string_length, "...");
      string_length += 3;
    } else if (available > 0) {
      strncpy(buffer + buffer_limit - 4, "...", available);
      string_length = buffer_limit - 1;
      buffer[string_length] = '\0';
    }

    // Remove trailing newline if present
    if (string_length > 0 && buffer[string_length - 1] == '\n') {
      string_length--;
      buffer[string_length] = '\0';
    }
  }

  return string_length;
}
