#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define __STDC_WANT_IEC_60559_FUNC_EXT__ 1
#define __STDC_WANT_IEC_60559_DFB_EXT__ 1
#include "../include/zor.h"
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

zor *zor_init(uint8_t rank, uint32_t *restrict shape) {
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

  tensor->data = zcalloc(tensor->data_size, *tensor->data);
  if (tensor->data == NULL) {
    LOG_ERROR("Memory allocation failed for tensor data array.");
    zfree(tensor->strides);
    zfree(tensor->shape);
    zfree(tensor);
    return NULL;
  }

  return tensor;
}

void zor_free(zor *restrict tensor) {
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

zor *zor_zeros(uint8_t rank, uint32_t *restrict shape) {
  return zor_init(rank, shape);
}

zor *zor_ones(uint8_t rank, uint32_t *restrict shape) {
  zor *tensor = zor_init(rank, shape);
  if (tensor == NULL) {
    return NULL;
  }

  for (uint64_t i = 0; i < tensor->data_size; i++) {
    tensor->data[i] = (zfl)1;
  }

  return tensor;
}

static bool pcg_is_seeded = false;

void zor_srandom(uint64_t seed) {
  if (!pcg_is_seeded)
    pcg_is_seeded = true;

  pcg32_srandom(seed, 54u);
}

zor *zor_random(uint8_t rank, uint32_t *restrict shape, zfl min, zfl max) {
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
    tensor->data[i] = (zfl)(min + width * (float)pcg32_random() / UINT32_MAX);
  }

  return tensor;
}

zor *zor_reshape(zor *restrict tensor, uint8_t rank, uint32_t *shape) {
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

zor *zor_transpose(zor *restrict tensor, int32_t *restrict axes) {
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
    t_shape[axes[i]] = tensor->shape[i];
    t_strides[axes[i]] = tensor->strides[i];
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
      LOG_ERROR("Offset out of bounds. Please contact developer.");
      zor_free(transpose);
      return NULL;
    }

    transpose->data[i] = tensor->data[offset];

    for (uint8_t i = 0; i < tensor->rank; i++) {
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

#define ELLIPSIS ((void *)-1)
zor *zor_slice(zor *tensor, uint32_t n_slice_triples,
               int32_t **restrict slice_triples) {
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

    for (uint8_t i = 0; i < tensor->rank; i++) {
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
zor_get_element(zor *restrict tensor, const int *restrict indices,
                float *restrict value) {
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

bool zor_set_element(zor *restrict tensor, const int *restrict indices,
                     float value) {
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
