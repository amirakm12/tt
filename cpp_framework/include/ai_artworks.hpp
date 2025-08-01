/**
 * AI-ARTWORKS Enterprise C++ Framework
 * High-Performance GPU-Accelerated Creative Suite
 * Copyright (c) 2025 AI-ARTWORKS Enterprise
 */

#ifndef AI_ARTWORKS_HPP
#define AI_ARTWORKS_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <chrono>

// Version information
#define AI_ARTWORKS_VERSION_MAJOR 1
#define AI_ARTWORKS_VERSION_MINOR 0
#define AI_ARTWORKS_VERSION_PATCH 0
#define AI_ARTWORKS_VERSION_STRING "1.0.0-enterprise"

// Platform detection
#ifdef _WIN32
    #define AI_ARTWORKS_PLATFORM_WINDOWS
    #ifdef AI_ARTWORKS_EXPORTS
        #define AI_ARTWORKS_API __declspec(dllexport)
    #else
        #define AI_ARTWORKS_API __declspec(dllimport)
    #endif
#else
    #define AI_ARTWORKS_API __attribute__((visibility("default")))
#endif

// Forward declarations
namespace ai_artworks {
    class Context;
    class Device;
    class CommandQueue;
    class Buffer;
    class Image;
    class Kernel;
    class Pipeline;
    class NeuralNetwork;
    class RenderEngine;
    class ComputeEngine;
}

// Core includes
#include "core/types.hpp"
#include "core/error.hpp"
#include "core/device.hpp"
#include "core/context.hpp"
#include "core/memory.hpp"
#include "core/command.hpp"

// Compute includes
#include "compute/kernel.hpp"
#include "compute/pipeline.hpp"
#include "compute/neural.hpp"

// Graphics includes
#include "graphics/renderer.hpp"
#include "graphics/shader.hpp"
#include "graphics/texture.hpp"
#include "graphics/mesh.hpp"

// AI includes
#include "ai/inference.hpp"
#include "ai/training.hpp"
#include "ai/optimization.hpp"

// Utils includes
#include "utils/profiler.hpp"
#include "utils/logger.hpp"
#include "utils/thread_pool.hpp"

namespace ai_artworks {

/**
 * Main entry point for AI-ARTWORKS framework
 */
class AI_ARTWORKS_API Framework {
public:
    /**
     * Initialize the framework
     * @param config Configuration parameters
     * @return Error code (0 for success)
     */
    static ErrorCode Initialize(const Config& config = Config());
    
    /**
     * Shutdown the framework
     */
    static void Shutdown();
    
    /**
     * Get framework version
     */
    static Version GetVersion();
    
    /**
     * Get available devices
     */
    static std::vector<std::shared_ptr<Device>> GetDevices();
    
    /**
     * Create a new context
     * @param device Target device
     * @return New context
     */
    static std::shared_ptr<Context> CreateContext(std::shared_ptr<Device> device);
    
    /**
     * Get the default context
     */
    static std::shared_ptr<Context> GetDefaultContext();
    
    /**
     * Enable profiling
     */
    static void EnableProfiling(bool enable = true);
    
    /**
     * Get profiling results
     */
    static ProfilingData GetProfilingData();
    
    /**
     * Set log level
     */
    static void SetLogLevel(LogLevel level);
    
    /**
     * Register error callback
     */
    static void SetErrorCallback(std::function<void(ErrorCode, const std::string&)> callback);

private:
    Framework() = delete;
    ~Framework() = delete;
    Framework(const Framework&) = delete;
    Framework& operator=(const Framework&) = delete;
};

} // namespace ai_artworks

#endif // AI_ARTWORKS_HPP