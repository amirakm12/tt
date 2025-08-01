/**
 * AI-ARTWORKS Enterprise C++ Framework
 * Main Framework Implementation
 */

#include "ai_artworks.hpp"
#include "core/internal.hpp"
#include <iostream>
#include <mutex>
#include <map>

// Platform-specific includes
#ifdef AI_ARTWORKS_PLATFORM_WINDOWS
    #include <windows.h>
    #include <d3d12.h>
    #include <dxgi1_6.h>
#endif

// CUDA includes
#ifdef AI_ARTWORKS_CUDA_ENABLED
    #include <cuda_runtime.h>
    #include <cudnn.h>
    #include <cublas_v2.h>
#endif

// Vulkan includes
#ifdef AI_ARTWORKS_VULKAN_ENABLED
    #include <vulkan/vulkan.h>
#endif

namespace ai_artworks {

// Internal framework state
namespace {
    struct FrameworkState {
        bool initialized = false;
        std::mutex mutex;
        std::vector<std::shared_ptr<Device>> devices;
        std::shared_ptr<Context> default_context;
        std::unique_ptr<Profiler> profiler;
        std::unique_ptr<Logger> logger;
        std::function<void(ErrorCode, const std::string&)> error_callback;
        Config config;
        
        // Performance counters
        std::atomic<uint64_t> total_operations{0};
        std::atomic<uint64_t> total_memory_allocated{0};
        std::atomic<uint64_t> gpu_time_ns{0};
    };
    
    FrameworkState g_state;
}

// Implementation of Framework methods
ErrorCode Framework::Initialize(const Config& config) {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    
    if (g_state.initialized) {
        return ErrorCode::AlreadyInitialized;
    }
    
    g_state.config = config;
    
    // Initialize logging
    g_state.logger = std::make_unique<Logger>(config.log_level);
    g_state.logger->Log(LogLevel::Info, "Initializing AI-ARTWORKS Framework v" AI_ARTWORKS_VERSION_STRING);
    
    // Initialize profiler
    if (config.enable_profiling) {
        g_state.profiler = std::make_unique<Profiler>();
    }
    
    // Detect and initialize devices
    ErrorCode err = DetectDevices();
    if (err != ErrorCode::Success) {
        g_state.logger->Log(LogLevel::Error, "Failed to detect devices");
        return err;
    }
    
    // Initialize CUDA if available
#ifdef AI_ARTWORKS_CUDA_ENABLED
    if (config.enable_cuda) {
        err = InitializeCUDA();
        if (err != ErrorCode::Success) {
            g_state.logger->Log(LogLevel::Warning, "CUDA initialization failed, falling back to CPU");
        }
    }
#endif
    
    // Initialize Vulkan if available
#ifdef AI_ARTWORKS_VULKAN_ENABLED
    if (config.enable_vulkan) {
        err = InitializeVulkan();
        if (err != ErrorCode::Success) {
            g_state.logger->Log(LogLevel::Warning, "Vulkan initialization failed");
        }
    }
#endif
    
    // Create default context
    if (!g_state.devices.empty()) {
        g_state.default_context = std::make_shared<Context>(g_state.devices[0]);
    }
    
    // Set up memory management
    MemoryManager::Initialize(config.memory_pool_size);
    
    // Initialize thread pool
    ThreadPool::Initialize(config.num_threads);
    
    g_state.initialized = true;
    g_state.logger->Log(LogLevel::Info, "Framework initialized successfully");
    
    return ErrorCode::Success;
}

void Framework::Shutdown() {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    
    if (!g_state.initialized) {
        return;
    }
    
    g_state.logger->Log(LogLevel::Info, "Shutting down AI-ARTWORKS Framework");
    
    // Clean up contexts
    g_state.default_context.reset();
    
    // Clean up devices
    g_state.devices.clear();
    
    // Shutdown subsystems
    ThreadPool::Shutdown();
    MemoryManager::Shutdown();
    
#ifdef AI_ARTWORKS_CUDA_ENABLED
    ShutdownCUDA();
#endif
    
#ifdef AI_ARTWORKS_VULKAN_ENABLED
    ShutdownVulkan();
#endif
    
    // Final cleanup
    g_state.profiler.reset();
    g_state.logger->Log(LogLevel::Info, "Framework shutdown complete");
    g_state.logger.reset();
    
    g_state.initialized = false;
}

Version Framework::GetVersion() {
    return Version{
        AI_ARTWORKS_VERSION_MAJOR,
        AI_ARTWORKS_VERSION_MINOR,
        AI_ARTWORKS_VERSION_PATCH,
        AI_ARTWORKS_VERSION_STRING
    };
}

std::vector<std::shared_ptr<Device>> Framework::GetDevices() {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    return g_state.devices;
}

std::shared_ptr<Context> Framework::CreateContext(std::shared_ptr<Device> device) {
    if (!g_state.initialized) {
        throw std::runtime_error("Framework not initialized");
    }
    
    return std::make_shared<Context>(device);
}

std::shared_ptr<Context> Framework::GetDefaultContext() {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    return g_state.default_context;
}

void Framework::EnableProfiling(bool enable) {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    
    if (enable && !g_state.profiler) {
        g_state.profiler = std::make_unique<Profiler>();
    } else if (!enable) {
        g_state.profiler.reset();
    }
}

ProfilingData Framework::GetProfilingData() {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    
    if (!g_state.profiler) {
        return ProfilingData{};
    }
    
    return g_state.profiler->GetData();
}

void Framework::SetLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    
    if (g_state.logger) {
        g_state.logger->SetLevel(level);
    }
}

void Framework::SetErrorCallback(std::function<void(ErrorCode, const std::string&)> callback) {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.error_callback = callback;
}

// Internal helper functions
ErrorCode DetectDevices() {
    g_state.devices.clear();
    
    // Detect CPU devices
    auto cpu_info = GetCPUInfo();
    auto cpu_device = std::make_shared<CPUDevice>(cpu_info);
    g_state.devices.push_back(cpu_device);
    
#ifdef AI_ARTWORKS_CUDA_ENABLED
    // Detect CUDA devices
    int cuda_device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&cuda_device_count);
    
    if (cuda_err == cudaSuccess && cuda_device_count > 0) {
        for (int i = 0; i < cuda_device_count; ++i) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            
            auto cuda_device = std::make_shared<CUDADevice>(i, props);
            g_state.devices.push_back(cuda_device);
            
            g_state.logger->Log(LogLevel::Info, 
                "Found CUDA device: " + std::string(props.name) + 
                " (Compute " + std::to_string(props.major) + "." + std::to_string(props.minor) + ")");
        }
    }
#endif
    
#ifdef AI_ARTWORKS_VULKAN_ENABLED
    // Detect Vulkan devices
    VkInstance instance = CreateVulkanInstance();
    if (instance) {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        
        if (device_count > 0) {
            std::vector<VkPhysicalDevice> physical_devices(device_count);
            vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());
            
            for (auto& phys_device : physical_devices) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(phys_device, &props);
                
                auto vulkan_device = std::make_shared<VulkanDevice>(phys_device, props);
                g_state.devices.push_back(vulkan_device);
                
                g_state.logger->Log(LogLevel::Info, 
                    "Found Vulkan device: " + std::string(props.deviceName));
            }
        }
        
        vkDestroyInstance(instance, nullptr);
    }
#endif
    
    g_state.logger->Log(LogLevel::Info, 
        "Detected " + std::to_string(g_state.devices.size()) + " compute devices");
    
    return g_state.devices.empty() ? ErrorCode::NoDeviceFound : ErrorCode::Success;
}

// Global error reporting
void ReportError(ErrorCode code, const std::string& message) {
    if (g_state.error_callback) {
        g_state.error_callback(code, message);
    }
    
    if (g_state.logger) {
        g_state.logger->Log(LogLevel::Error, message);
    }
}

// Performance tracking
void TrackOperation(const std::string& op_name, uint64_t duration_ns) {
    g_state.total_operations++;
    g_state.gpu_time_ns += duration_ns;
    
    if (g_state.profiler) {
        g_state.profiler->RecordOperation(op_name, duration_ns);
    }
}

void TrackMemoryAllocation(size_t bytes) {
    g_state.total_memory_allocated += bytes;
    
    if (g_state.profiler) {
        g_state.profiler->RecordMemoryAllocation(bytes);
    }
}

} // namespace ai_artworks