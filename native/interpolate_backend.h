#ifndef INTERPOLATE_BACKEND_H
#define INTERPOLATE_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define INTERPOLATE_BACKEND_EXPORT __declspec(dllexport)
#else
#define INTERPOLATE_BACKEND_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct interpolate_backend interpolate_backend;

typedef struct interpolate_backend_configuration {
    int32_t gpu_index;
    int32_t cpu_thread_count;
    int32_t use_uhd_mode;
} interpolate_backend_configuration;

enum interpolate_backend_status {
    INTERPOLATE_BACKEND_STATUS_OK = 0,
    INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT = 1,
    INTERPOLATE_BACKEND_STATUS_UNAVAILABLE = 2,
    INTERPOLATE_BACKEND_STATUS_MODEL_ERROR = 3,
    INTERPOLATE_BACKEND_STATUS_INFERENCE_ERROR = 4,
    INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR = 5
};

INTERPOLATE_BACKEND_EXPORT uint32_t interpolate_backend_abi_version(void);

INTERPOLATE_BACKEND_EXPORT int32_t interpolate_backend_gpu_count(
    char *error_message,
    size_t error_message_size);

INTERPOLATE_BACKEND_EXPORT int32_t interpolate_backend_gpu_name(
    int32_t gpu_index,
    char *gpu_name,
    size_t gpu_name_size,
    char *error_message,
    size_t error_message_size);

INTERPOLATE_BACKEND_EXPORT int32_t interpolate_backend_create(
    const interpolate_backend_configuration *configuration,
    const char *model_directory,
    size_t model_directory_size,
    interpolate_backend **backend_out,
    char *error_message,
    size_t error_message_size);

INTERPOLATE_BACKEND_EXPORT int32_t interpolate_backend_process_rgb24(
    interpolate_backend *backend,
    const uint8_t *frame_before,
    size_t frame_before_size,
    const uint8_t *frame_after,
    size_t frame_after_size,
    uint32_t width,
    uint32_t height,
    size_t row_stride,
    float timestep,
    uint8_t *frame_output,
    size_t output_size,
    char *error_message,
    size_t error_message_size);

INTERPOLATE_BACKEND_EXPORT void interpolate_backend_destroy(
    interpolate_backend *backend);

INTERPOLATE_BACKEND_EXPORT void interpolate_backend_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
