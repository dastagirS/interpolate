#include "interpolate_backend.h"

#include "rife/rife.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>

#include "gpu.h"

namespace {
constexpr uint32_t backend_abi_version = 2;
constexpr uint64_t backend_magic = 0x4952504C52494645ULL;
constexpr int32_t cpu_thread_count_min = 1;
constexpr int32_t cpu_thread_count_max = 16;
constexpr size_t model_path_size_max = 4096;
constexpr size_t rgb_channel_count = 3;

std::mutex gpu_mutex;
bool gpu_instance_initialized = false;
size_t backend_count = 0;

void clear_error(char *error_message, size_t error_message_size)
{
    static_assert(sizeof(char) == 1);
    assert(INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT > INTERPOLATE_BACKEND_STATUS_OK);
    if (error_message == nullptr || error_message_size == 0)
        return;
    error_message[0] = '\0';
    assert(error_message[0] == '\0');
    assert(error_message_size > 0);
}

void write_error(char *error_message, size_t error_message_size, const char *message)
{
    assert(message != nullptr);
    assert(INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR > INTERPOLATE_BACKEND_STATUS_OK);
    const char *resolved_message = message[0] == '\0' ? "native backend failed" : message;
    if (error_message == nullptr || error_message_size == 0)
        return;
    std::snprintf(error_message, error_message_size, "%s", resolved_message);
    assert(error_message[error_message_size - 1] == '\0' || std::strlen(error_message) < error_message_size);
    assert(std::strlen(error_message) < error_message_size);
}

void ensure_gpu_instance_locked()
{
    assert(!gpu_instance_initialized || ncnn::get_gpu_count() >= 0);
    assert(backend_count < std::numeric_limits<size_t>::max());
    if (!gpu_instance_initialized)
    {
        ncnn::create_gpu_instance();
        gpu_instance_initialized = true;
    }
    assert(gpu_instance_initialized);
    assert(ncnn::get_gpu_count() >= 0);
}

void release_gpu_instance_if_idle_locked()
{
    assert(gpu_instance_initialized || backend_count == 0);
    assert(backend_count < std::numeric_limits<size_t>::max());
    if (gpu_instance_initialized && backend_count == 0)
    {
        ncnn::destroy_gpu_instance();
        gpu_instance_initialized = false;
    }
    assert(backend_count != 0 || !gpu_instance_initialized);
    assert(!gpu_instance_initialized || ncnn::get_gpu_count() >= 0);
}

bool model_files_exist(const std::filesystem::path &model_directory)
{
    assert(!model_directory.empty());
    assert(model_directory.native().size() <= model_path_size_max);
    std::error_code error;
    const auto parameter_path = model_directory / "flownet.param";
    const auto weights_path = model_directory / "flownet.bin";
    const bool parameter_valid = std::filesystem::is_regular_file(parameter_path, error) && !error;
    error.clear();
    const bool weights_valid = std::filesystem::is_regular_file(weights_path, error) && !error;
    assert(!parameter_valid || !parameter_path.empty());
    assert(!weights_valid || !weights_path.empty());
    return parameter_valid && weights_valid;
}
}

struct interpolate_backend {
    uint64_t magic;
    std::unique_ptr<RIFE> engine;
};

extern "C" uint32_t interpolate_backend_abi_version(void)
{
    static_assert(backend_abi_version > 0);
    static_assert(sizeof(uint32_t) == 4);
    const uint32_t version = backend_abi_version;
    assert(version == backend_abi_version);
    assert(version != 0);
    return version;
}

extern "C" int32_t interpolate_backend_gpu_count(char *error_message, size_t error_message_size)
{
    assert(backend_abi_version > 0);
    assert(INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR > INTERPOLATE_BACKEND_STATUS_OK);
    clear_error(error_message, error_message_size);
    try
    {
        std::lock_guard<std::mutex> lock(gpu_mutex);
        ensure_gpu_instance_locked();
        const int count = ncnn::get_gpu_count();
        assert(count >= 0);
        assert(count <= std::numeric_limits<int32_t>::max());
        return static_cast<int32_t>(count);
    }
    catch (...)
    {
        write_error(error_message, error_message_size, "failed to enumerate Vulkan devices");
        assert(backend_abi_version > 0);
        assert(INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR > 0);
        return -INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR;
    }
}

extern "C" int32_t interpolate_backend_gpu_name(
    int32_t gpu_index,
    char *gpu_name,
    size_t gpu_name_size,
    char *error_message,
    size_t error_message_size)
{
    assert(backend_abi_version > 0);
    assert(INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT > INTERPOLATE_BACKEND_STATUS_OK);
    clear_error(error_message, error_message_size);
    if (gpu_name == nullptr || gpu_name_size < 2 || gpu_index < 0)
    {
        write_error(error_message, error_message_size, "invalid GPU name arguments");
        return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
    }
    gpu_name[0] = '\0';

    try
    {
        std::lock_guard<std::mutex> lock(gpu_mutex);
        ensure_gpu_instance_locked();
        const int gpu_count = ncnn::get_gpu_count();
        if (gpu_index >= gpu_count)
        {
            write_error(error_message, error_message_size, "GPU index is out of range");
            return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
        }
        const char *name = ncnn::get_gpu_info(gpu_index).device_name();
        std::snprintf(gpu_name, gpu_name_size, "%s", name == nullptr ? "Unknown Vulkan GPU" : name);
        assert(gpu_name[0] != '\0');
        assert(gpu_index < gpu_count);
        return INTERPOLATE_BACKEND_STATUS_OK;
    }
    catch (...)
    {
        write_error(error_message, error_message_size, "failed to query Vulkan device");
        assert(backend_abi_version > 0);
        assert(INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR > 0);
        return INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR;
    }
}

extern "C" int32_t interpolate_backend_create(
    const interpolate_backend_configuration *configuration,
    const char *model_directory,
    size_t model_directory_size,
    interpolate_backend **backend_out,
    char *error_message,
    size_t error_message_size)
{
    assert(backend_abi_version > 0);
    assert(model_path_size_max > 0);
    clear_error(error_message, error_message_size);
    if (backend_out == nullptr || configuration == nullptr || model_directory == nullptr || model_directory_size == 0 || model_directory_size > model_path_size_max)
    {
        write_error(error_message, error_message_size, "invalid backend configuration");
        return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
    }
    *backend_out = nullptr;
    if (configuration->cpu_thread_count < cpu_thread_count_min || configuration->cpu_thread_count > cpu_thread_count_max || configuration->use_uhd_mode < 0 || configuration->use_uhd_mode > 1)
    {
        write_error(error_message, error_message_size, "native thread count or UHD setting is invalid");
        return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
    }

    try
    {
        if (std::find(model_directory, model_directory + model_directory_size, '\0') != model_directory + model_directory_size)
        {
            write_error(error_message, error_message_size, "model directory contains an embedded null byte");
            return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
        }
        const std::string model_directory_string(model_directory, model_directory_size);
        const std::filesystem::path model_path(model_directory_string);
        if (!model_files_exist(model_path))
        {
            write_error(error_message, error_message_size, "RIFE model files are missing");
            return INTERPOLATE_BACKEND_STATUS_MODEL_ERROR;
        }

        std::lock_guard<std::mutex> lock(gpu_mutex);
        if (backend_count != 0)
        {
            write_error(error_message, error_message_size, "only one native backend instance is supported");
            return INTERPOLATE_BACKEND_STATUS_UNAVAILABLE;
        }
        ensure_gpu_instance_locked();
        const int gpu_count = ncnn::get_gpu_count();
        if (configuration->gpu_index < -1 || configuration->gpu_index >= gpu_count)
        {
            release_gpu_instance_if_idle_locked();
            write_error(error_message, error_message_size, "configured Vulkan device is unavailable");
            return INTERPOLATE_BACKEND_STATUS_UNAVAILABLE;
        }

        auto backend = std::make_unique<interpolate_backend>();
        backend->magic = backend_magic;
        backend->engine = std::make_unique<RIFE>(
            configuration->gpu_index,
            false,
            false,
            configuration->use_uhd_mode != 0,
            configuration->cpu_thread_count,
            false,
            true,
            64);
        const int load_status = backend->engine->load(model_directory_string);
        if (load_status != 0)
        {
            backend->engine.reset();
            release_gpu_instance_if_idle_locked();
            write_error(error_message, error_message_size, "failed to load RIFE 4.25 model");
            return INTERPOLATE_BACKEND_STATUS_MODEL_ERROR;
        }

        backend_count = 1;
        *backend_out = backend.release();
        assert(*backend_out != nullptr);
        assert((*backend_out)->magic == backend_magic);
        return INTERPOLATE_BACKEND_STATUS_OK;
    }
    catch (const std::exception &error)
    {
        std::lock_guard<std::mutex> lock(gpu_mutex);
        release_gpu_instance_if_idle_locked();
        write_error(error_message, error_message_size, error.what());
        assert(*backend_out == nullptr);
        assert(backend_abi_version > 0);
        return INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR;
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(gpu_mutex);
        release_gpu_instance_if_idle_locked();
        write_error(error_message, error_message_size, "unknown native backend initialization error");
        assert(*backend_out == nullptr);
        assert(backend_abi_version > 0);
        return INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR;
    }
}

extern "C" int32_t interpolate_backend_process_rgb24(
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
    size_t error_message_size)
{
    assert(backend_abi_version > 0);
    assert(rgb_channel_count == 3);
    clear_error(error_message, error_message_size);
    if (backend == nullptr || backend->magic != backend_magic || backend->engine == nullptr || frame_before == nullptr || frame_after == nullptr || frame_output == nullptr || width == 0 || height == 0 || !std::isfinite(timestep) || timestep <= 0.0F || timestep >= 1.0F)
    {
        write_error(error_message, error_message_size, "invalid interpolation arguments");
        return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
    }
    if (width > std::numeric_limits<size_t>::max() / rgb_channel_count)
    {
        write_error(error_message, error_message_size, "frame width is too large");
        return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
    }
    const size_t expected_stride = static_cast<size_t>(width) * rgb_channel_count;
    if (row_stride != expected_stride || height > std::numeric_limits<size_t>::max() / row_stride)
    {
        write_error(error_message, error_message_size, "RGB24 frame stride is invalid");
        return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
    }
    const size_t expected_size = row_stride * static_cast<size_t>(height);
    if (frame_before_size < expected_size || frame_after_size < expected_size || output_size < expected_size)
    {
        write_error(error_message, error_message_size, "an RGB24 frame buffer is too small");
        return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
    }
    if (width > static_cast<uint32_t>(std::numeric_limits<int>::max()) || height > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        write_error(error_message, error_message_size, "RGB24 frame dimensions exceed native limits");
        return INTERPOLATE_BACKEND_STATUS_INVALID_ARGUMENT;
    }

    try
    {
        ncnn::Mat before(static_cast<int>(width), static_cast<int>(height), const_cast<uint8_t *>(frame_before), rgb_channel_count, rgb_channel_count);
        ncnn::Mat after(static_cast<int>(width), static_cast<int>(height), const_cast<uint8_t *>(frame_after), rgb_channel_count, rgb_channel_count);
        ncnn::Mat output(static_cast<int>(width), static_cast<int>(height), frame_output, rgb_channel_count, rgb_channel_count);
        const int process_status = backend->engine->process(before, after, timestep, output);
        if (process_status != 0)
        {
            write_error(error_message, error_message_size, "RIFE inference failed");
            return INTERPOLATE_BACKEND_STATUS_INFERENCE_ERROR;
        }
        assert(output.data == frame_output);
        assert(output_size >= expected_size);
        return INTERPOLATE_BACKEND_STATUS_OK;
    }
    catch (const std::exception &error)
    {
        write_error(error_message, error_message_size, error.what());
        assert(backend->magic == backend_magic);
        assert(backend_abi_version > 0);
        return INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR;
    }
    catch (...)
    {
        write_error(error_message, error_message_size, "unknown RIFE inference error");
        assert(backend->magic == backend_magic);
        assert(backend_abi_version > 0);
        return INTERPOLATE_BACKEND_STATUS_INTERNAL_ERROR;
    }
}

extern "C" void interpolate_backend_destroy(interpolate_backend *backend)
{
    assert(backend == nullptr || backend->magic == backend_magic);
    assert(backend_count <= 1);
    if (backend == nullptr)
        return;

    std::lock_guard<std::mutex> lock(gpu_mutex);
    backend->engine.reset();
    backend->magic = 0;
    delete backend;
    if (backend_count > 0)
        backend_count--;
    release_gpu_instance_if_idle_locked();
    assert(backend_count == 0);
    assert(!gpu_instance_initialized);
}

extern "C" void interpolate_backend_shutdown(void)
{
    assert(backend_count <= 1);
    assert(!gpu_instance_initialized || ncnn::get_gpu_count() >= 0);
    std::lock_guard<std::mutex> lock(gpu_mutex);
    release_gpu_instance_if_idle_locked();
    assert(backend_count != 0 || !gpu_instance_initialized);
    assert(backend_count <= 1);
}
