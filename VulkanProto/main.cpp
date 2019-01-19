﻿/*
* Vulkan Windowed Program
*
* Copyright (C) 2016 Valve Corporation
* Copyright (C) 2016 LunarG, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/*
Vulkan C++ Windowed Project Template
Create and destroy a Vulkan surface on an SDL window.
*/

// Enable the WSI extensions
#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

// Tell SDL not to mess with main()
#define SDL_MAIN_HANDLED

#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>

#include <vulkan/vulkan.hpp>
#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>

#include <iostream>
#include <vector>
#include <sstream>

#undef min
#undef max

#ifdef WIN32
// Class for redirecting cout to Visual Studio output window
class dbg_stream_for_cout : public std::stringbuf
{
public:
  ~dbg_stream_for_cout() { sync(); }
  int sync()
  {
    ::OutputDebugStringA(str().c_str());
    str(std::string()); // Clear the string buffer
    return 0;
  }
};
dbg_stream_for_cout g_DebugStreamFor_cout;
#endif

struct DepthData
{
  vk::Image image;
  vk::DeviceMemory mem;
  vk::ImageView view;
  vk::Format format;
};

struct SwapchainBuffers
{
  vk::Image image;
  vk::CommandBuffer cmd;
  vk::CommandBuffer graphics_to_present_cmd;
  vk::ImageView view;
};

struct UniformData
{
  VkBuffer buf;
  VkDeviceMemory mem;
  VkDescriptorBufferInfo bufferInfo;
};

struct vkinfo
{
  std::vector<vk::PhysicalDevice> gpus;
  vk::PhysicalDevice gpu;
  vk::Device device;
  vk::SurfaceKHR surface;
  vk::SurfaceCapabilitiesKHR surfaceCapabilities;
  vk::Instance inst;

  vk::CommandPool cmdPool;
  std::vector<vk::CommandBuffer> cmdBuffers;

  uint32_t gfxQueueFamilyIdx{ 0 }, prsntQueueFamilyIdx{ 0 };
  vk::Queue gfxQueue;
  vk::Queue prsntQueue;

  vk::SwapchainKHR swapChain;
  vk::Format surfaceFormat;
  std::vector<SwapchainBuffers> swapchainBuffers;

  vk::PhysicalDeviceMemoryProperties memoryProperties;
  vk::PhysicalDeviceProperties deviceProperties;

  DepthData depth;

  std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
  vk::PipelineLayout pipelineLayout;

  UniformData uniformData;
  vk::DescriptorPool descriptorPool;
  std::vector<vk::DescriptorSet> descriptorSets;

  vk::RenderPass renderPass;


} info;

struct sdlinfo
{
  SDL_Window* window;
} sdlinfo;

vk::PresentModeKHR getPresentMode(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& gpu)
{
  auto modes = gpu.getSurfacePresentModesKHR(surface);
  if(modes.size() == 1)
    return modes.at(0);

  for(const auto& mode : modes)
    if(mode == vk::PresentModeKHR::eMailbox)
      return mode;

  return vk::PresentModeKHR::eFifo;
}

vk::SurfaceKHR createVulkanSurface(const vk::Instance& instance, SDL_Window* window);
std::vector<const char*> getAvailableWSIExtensions();

bool memory_type_from_properties(uint32_t typeBits,
                                 vk::MemoryPropertyFlags requirements_mask,
                                 uint32_t& typeIndex)
{
  // Search memtypes to find first index with those properties
  for(uint32_t i = 0; i < info.memoryProperties.memoryTypeCount; ++i) {
    if((typeBits & 1) == 1) {
      // Type is available, does it match user properties?
      if((info.memoryProperties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
        typeIndex = i;
        return true;
      }
    }
    typeBits >>= 1;
  }
  // No memory types matched, return failure
  return false;
}

int setupApplicationAndInstance()
{
  // Use validation layers if this is a debug build, and use WSI extensions regardless
  std::vector<const char*> extensions = getAvailableWSIExtensions();
  std::vector<const char*> layers;
  //#if defined(_DEBUG)
  layers.push_back("VK_LAYER_LUNARG_standard_validation");
  //#endif

  // vk::ApplicationInfo allows the programmer to specifiy some basic information about the
  // program, which can be useful for layers and tools to provide more debug information.
  vk::ApplicationInfo appInfo = vk::ApplicationInfo()
    .setPApplicationName("Peppes ultimata Vulkan-program")
    .setApplicationVersion(1)
    .setPEngineName("Peppes Motor")
    .setEngineVersion(1)
    .setApiVersion(VK_API_VERSION_1_0);

  // vk::InstanceCreateInfo is where the programmer specifies the layers and/or extensions that
  // are needed.
  vk::InstanceCreateInfo instInfo = vk::InstanceCreateInfo()
    .setFlags(vk::InstanceCreateFlags())
    .setPApplicationInfo(&appInfo)
    .setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()))
    .setPpEnabledExtensionNames(extensions.data())
    .setEnabledLayerCount(static_cast<uint32_t>(layers.size()))
    .setPpEnabledLayerNames(layers.data());

  // Create the Vulkan instance.
  vk::Instance instance;
  try {
    instance = vk::createInstance(instInfo);
  }
  catch(const std::exception& e) {
    std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
    return 1;
  }
  info.inst = instance;

  // Create an SDL window that supports Vulkan and OpenGL rendering.
  if(SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::cout << "Could not initialize SDL." << std::endl;
    return 1;
  }
  SDL_Window* window = SDL_CreateWindow("Vulkan Window", SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL);
  if(window == NULL) {
    std::cout << "Could not create SDL window." << std::endl;
    return 1;
  }
  sdlinfo.window = window;

  // Create a platform specific Vulkan surface for rendering
  vk::SurfaceKHR surface;
  try {
    surface = createVulkanSurface(instance, window);
  }
  catch(const std::exception& e) {
    std::cout << "Failed to create Vulkan surface: " << e.what() << std::endl;
    instance.destroy();
    return 1;
  }
  info.surface = surface;

  return 0;
}

int setupDevicesAndQueues()
{
  // Enumerate devices
  try {
    info.gpus = info.inst.enumeratePhysicalDevices();
    info.gpu = info.gpus[0];
  }
  catch(const std::exception& e) {
    std::cout << "No physical devices found: " << e.what() << std::endl;
    info.inst.destroy();
    return 1;
  }

  // Setup the gpu memory properties
  info.memoryProperties = info.gpu.getMemoryProperties();
  info.deviceProperties = info.gpu.getProperties();

  /* Call with nullptr data to get count */
  auto queueFamilyProps = info.gpu.getQueueFamilyProperties();

  // Create the device and queues
  // Iterate over each queue to learn whether it supports presenting:
  std::vector<vk::Bool32> supportsPresent(queueFamilyProps.size());
  for(uint32_t i = 0; i < queueFamilyProps.size(); ++i) {
    info.gpu.getSurfaceSupportKHR(i, info.surface, &supportsPresent[i]);
  }

  uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
  uint32_t presentQueueFamilyIndex = UINT32_MAX;
  for(uint32_t i = 0; i < queueFamilyProps.size(); ++i) {
    if(queueFamilyProps[i].queueFlags & vk::QueueFlagBits::eGraphics) {
      if(graphicsQueueFamilyIndex == UINT32_MAX) {
        graphicsQueueFamilyIndex = i;
      }

      if(supportsPresent[i] == VK_TRUE) {
        graphicsQueueFamilyIndex = i;
        presentQueueFamilyIndex = i;
        break;
      }
    }
  }

  if(presentQueueFamilyIndex == UINT32_MAX) {
    // If didn't find a queue that supports both graphics and present,
    // then find a separate present queue.
    for(uint32_t i = 0; i < queueFamilyProps.size(); ++i) {
      if(supportsPresent[i] == VK_TRUE) {
        presentQueueFamilyIndex = i;
        break;
      }
    }
  }

  // Generate error if could not find both a graphics and a present queue
  if(graphicsQueueFamilyIndex == UINT32_MAX || presentQueueFamilyIndex == UINT32_MAX) {
    std::cout << "Could not find both graphics and present queues" << std::endl;
    info.inst.destroy();
    return 1;
  }

  info.gfxQueueFamilyIdx = graphicsQueueFamilyIndex;
  info.prsntQueueFamilyIdx = presentQueueFamilyIndex;

  // Setup queue creation info
  float queuePriorities[1] = { 0.0f };
  auto queueInfo = vk::DeviceQueueCreateInfo()
    .setQueueCount(1)
    .setPQueuePriorities(queuePriorities)
    .setQueueFamilyIndex(graphicsQueueFamilyIndex);

  const std::vector<const char*> extensionNames = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
  const auto createInfo = vk::DeviceCreateInfo()
    .setPQueueCreateInfos(&queueInfo)
    .setQueueCreateInfoCount(1)
    .setEnabledExtensionCount(1)
    .setPpEnabledExtensionNames(extensionNames.data());

  try {
    info.device = info.gpus[0].createDevice(createInfo);
  }
  catch(const std::exception& e) {
    std::cout << "Unable to create vkDevice: " << e.what() << std::endl;
    info.inst.destroy();
    return 1;
  }

  // Save handles to the Queues
  info.gfxQueue = info.device.getQueue(graphicsQueueFamilyIndex, 0);
  info.prsntQueue = info.device.getQueue(presentQueueFamilyIndex, 0);

  // Setup and create a command queue pool
  const auto cmdPoolInfo = vk::CommandPoolCreateInfo().setQueueFamilyIndex(graphicsQueueFamilyIndex);
  info.cmdPool = info.device.createCommandPool(cmdPoolInfo);
  // Allocate the command buffers for that pool
  const auto cmdBufAllocInfo = vk::CommandBufferAllocateInfo()
    .setCommandBufferCount(1)
    .setCommandPool(info.cmdPool);
  info.cmdBuffers = info.device.allocateCommandBuffers(cmdBufAllocInfo);

  return 0;
}

int setupSwapChains()
{
  // Create Swapchains for our platform specific surface
  const auto oldSwapChain = info.swapChain;
  const auto formats = info.gpu.getSurfaceFormatsKHR(info.surface);
  const auto pSurfCap = info.gpu.getSurfaceCapabilitiesKHR(info.surface);
  auto swapchainCreateInfo = vk::SwapchainCreateInfoKHR()
    .setSurface(info.surface)
    .setImageFormat(formats.at(0).format)
    .setImageColorSpace(formats.at(0).colorSpace)
    .setMinImageCount(std::max(pSurfCap.maxImageCount, pSurfCap.minImageCount + 1))
    .setImageArrayLayers(1)
    .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
    .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
    .setClipped(true)
    .setImageExtent(pSurfCap.currentExtent)
    .setPreTransform(pSurfCap.currentTransform)
    .setPresentMode(getPresentMode(info.surface, info.gpu))
    .setImageSharingMode(vk::SharingMode::eExclusive)
    .setQueueFamilyIndexCount(0)
    .setPQueueFamilyIndices(nullptr)
    .setOldSwapchain(oldSwapChain);

  // Save surface format for later use
  info.surfaceFormat = formats.front().format;

  // If indices are separate for gfx and present, we need to handle it
  if(info.gfxQueue != info.prsntQueue)
  {
    std::vector<uint32_t> indices = { info.gfxQueueFamilyIdx, info.prsntQueueFamilyIdx };
    swapchainCreateInfo.setImageSharingMode(vk::SharingMode::eConcurrent)
      .setQueueFamilyIndexCount((uint32_t)indices.size())
      .setPQueueFamilyIndices(indices.data());
  }
  info.surfaceCapabilities = pSurfCap;

  // Create the swapchain, clear old one, and create images and imageviews
  info.swapChain = info.device.createSwapchainKHR(swapchainCreateInfo);
  if(oldSwapChain)
    info.device.destroySwapchainKHR(oldSwapChain);

  auto swapChainImages = info.device.getSwapchainImagesKHR(info.swapChain);
  info.swapchainBuffers.resize(swapChainImages.size());
  auto imageViewCreateInfo = vk::ImageViewCreateInfo()
    .setViewType(vk::ImageViewType::e2D)
    .setFormat(formats.at(0).format)
    .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

  // Initialize the images using the same info
  for(uint32_t i = 0u; i < swapChainImages.size(); ++i)
  {
    imageViewCreateInfo.setImage(swapChainImages[i]);
    info.swapchainBuffers[i].image = swapChainImages[i];
    info.swapchainBuffers[i].view = info.device.createImageView(imageViewCreateInfo);
  }

  return 0;
}

int setupDepth()
{
  // Setup depth buffer
  info.depth.format = vk::Format::eD16Unorm;
  const auto depthImageInfo = vk::ImageCreateInfo()
    .setFormat(info.depth.format)
    .setImageType(vk::ImageType::e2D)
    .setExtent({ info.surfaceCapabilities.currentExtent.width, info.surfaceCapabilities.currentExtent.height, 1 })
    .setMipLevels(1)
    .setArrayLayers(1)
    .setSamples(vk::SampleCountFlagBits::e1)
    .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment);
  info.depth.image = info.device.createImage(depthImageInfo);

  // We need memory requirements from the GPU to allocate depth buffer memory
  auto memReqs = info.device.getImageMemoryRequirements(info.depth.image);
  auto memAllocInfo = vk::MemoryAllocateInfo()
    .setAllocationSize(memReqs.size)
    .setMemoryTypeIndex(0);
  const auto pass = memory_type_from_properties(memReqs.memoryTypeBits,
                                                vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                memAllocInfo.memoryTypeIndex);
  assert(pass);
  info.depth.mem = info.device.allocateMemory(memAllocInfo);
  // Bind the memory to the depth buffer
  info.device.bindImageMemory(info.depth.image, info.depth.mem, 0);

  // Create the Image View for the Depth Buffer, to describe how to use it
  const auto depthBufferImageViewInfo = vk::ImageViewCreateInfo()
    .setFormat(info.depth.format)
    .setImage(info.depth.image)
    .setViewType(vk::ImageViewType::e2D)
    .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
  info.depth.view = info.device.createImageView(depthBufferImageViewInfo);

  return 0;
}

int setupUniformBuffer(void* data, size_t size)
{
  const auto buf_info = vk::BufferCreateInfo()
    .setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
    .setSize(size);
  const auto buffer = info.device.createBuffer(buf_info); // Create buffer object

  // We need memory requirements from the GPU to allocate uniform buffer memory
  auto memReqs = info.device.getBufferMemoryRequirements(buffer);
  auto memAllocInfo = vk::MemoryAllocateInfo()
    .setAllocationSize(memReqs.size)
    .setMemoryTypeIndex(0);
  const auto pass = memory_type_from_properties(memReqs.memoryTypeBits,
                                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                memAllocInfo.memoryTypeIndex);
  const auto memory = info.device.allocateMemory(memAllocInfo);

  // Write the actual uniform data to the buffer
  const auto buf_data = info.device.mapMemory(memory, 0, memReqs.size);
  memcpy(buf_data, data, size);
  info.device.unmapMemory(memory); // Unmap the memory buffer ASAP
  info.device.bindBufferMemory(buffer, memory, 0); // Associate the allocated memory with the buffer object

  // Save buffer info for later access
  info.uniformData.buf = buffer;
  info.uniformData.bufferInfo.buffer = info.uniformData.buf;
  info.uniformData.bufferInfo.offset = 0;
  info.uniformData.bufferInfo.range = size;

  return 0;
}

int setupDescriptorSetLayoutAndPipelineLayout()
{
  // Number of descriptor sets
  const auto numOfDescriptors = 1;
  // Create a layout binding for a uniform buffer used in vertex shader
  const auto layoutBinding = vk::DescriptorSetLayoutBinding()
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(numOfDescriptors)
    .setStageFlags(vk::ShaderStageFlagBits::eVertex);
  const auto descriptorLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo()
    .setBindingCount(1)
    .setPBindings(&layoutBinding);
  // Create descriptorLayout and save it in info object
  const auto descriptorSetLayout = info.device.createDescriptorSetLayout(descriptorLayoutCreateInfo);
  info.descriptorSetLayouts.push_back(descriptorSetLayout);

  const auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo()
    .setSetLayoutCount(numOfDescriptors)
    .setPSetLayouts(&descriptorSetLayout);
  info.pipelineLayout = info.device.createPipelineLayout(pipelineLayoutCreateInfo);

  /*
  In GLSL:
   layout (set=M, binding=N) uniform sampler2D variableNameArray[I];

    M refers the the M'th descriptor set layout in the pSetLayouts member of the pipeline layout
    N refers to the N'th descriptor set (binding) in M's pBindings member of the descriptor set layout
    I is the index into the array of descriptors in N's descriptor set
  */

  return 0;
}

int setupDescriptorSetPool()
{
  const auto typeCount = vk::DescriptorPoolSize()
    .setType(vk::DescriptorType::eUniformBuffer)
    .setDescriptorCount(1);
  const auto descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo()
    .setMaxSets(1)
    .setPoolSizeCount(1)
    .setPPoolSizes(&typeCount);
  info.descriptorPool = info.device.createDescriptorPool(descriptorPoolCreateInfo);

  return 0;
}

int allocateDescriptorSet()
{
  const auto allocInfo = vk::DescriptorSetAllocateInfo()
    .setDescriptorPool(info.descriptorPool)
    .setDescriptorSetCount(info.descriptorSetLayouts.size())
    .setPSetLayouts(info.descriptorSetLayouts.data());
  info.descriptorSets = info.device.allocateDescriptorSets(allocInfo);

  const auto writes = std::vector<vk::WriteDescriptorSet>{ vk::WriteDescriptorSet()
    .setDstSet(info.descriptorSets.front())
    .setDescriptorCount(info.descriptorSets.size())
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
    .setPBufferInfo((const vk::DescriptorBufferInfo*)&info.uniformData.bufferInfo) };
  // Copy the VkDescriptorBufferInfo into the descriptor (the device)
  info.device.updateDescriptorSets(writes, nullptr);

  return 0;
}

int initRenderPass()
{
  // Set up two attachments
  const auto attachments = std::vector<vk::AttachmentDescription>
  {
    // The framebuffer (color) attachment
    vk::AttachmentDescription()
      .setFormat(info.surfaceFormat)
      .setSamples(vk::SampleCountFlagBits::e1)
      .setLoadOp(vk::AttachmentLoadOp::eClear)
      .setStoreOp(vk::AttachmentStoreOp::eStore)
      .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
      .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
      .setInitialLayout(vk::ImageLayout::eUndefined)
      .setFinalLayout(vk::ImageLayout::ePresentSrcKHR)
    ,
    // The depth attachment
    vk::AttachmentDescription()
    .setFormat(info.depth.format)
    .setSamples(vk::SampleCountFlagBits::e1)
    .setLoadOp(vk::AttachmentLoadOp::eClear)
    .setStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
    .setInitialLayout(vk::ImageLayout::eUndefined)
    .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
  };

  // AttachmentReferences
  const auto colorReference = vk::AttachmentReference()
    .setAttachment(0) // Matches the array index above
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);
  const auto depthReference = vk::AttachmentReference()
    .setAttachment(1)
    .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

  // Subpass description
  const auto subpass = vk::SubpassDescription()
    .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics) // Graphics or compute
    .setColorAttachmentCount(1)
    .setPColorAttachments(&colorReference)
    .setPDepthStencilAttachment(&depthReference);

  // Define the render pass
  const auto renderPassCreateInfo = vk::RenderPassCreateInfo()
    .setAttachmentCount(attachments.size())
    .setPAttachments(attachments.data())
    .setSubpassCount(1)
    .setPSubpasses(&subpass);
  info.renderPass = info.device.createRenderPass(renderPassCreateInfo);

  return 0;
}

EShLanguage FindLanguage(const VkShaderStageFlagBits shader_type)
{
  switch(shader_type) {
  case VK_SHADER_STAGE_VERTEX_BIT:
    return EShLangVertex;

  case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
    return EShLangTessControl;

  case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
    return EShLangTessEvaluation;

  case VK_SHADER_STAGE_GEOMETRY_BIT:
    return EShLangGeometry;

  case VK_SHADER_STAGE_FRAGMENT_BIT:
    return EShLangFragment;

  case VK_SHADER_STAGE_COMPUTE_BIT:
    return EShLangCompute;

  default:
    return EShLangVertex;
  }
}

void init_resources(TBuiltInResource &Resources)
{
  Resources.maxLights = 32;
  Resources.maxClipPlanes = 6;
  Resources.maxTextureUnits = 32;
  Resources.maxTextureCoords = 32;
  Resources.maxVertexAttribs = 64;
  Resources.maxVertexUniformComponents = 4096;
  Resources.maxVaryingFloats = 64;
  Resources.maxVertexTextureImageUnits = 32;
  Resources.maxCombinedTextureImageUnits = 80;
  Resources.maxTextureImageUnits = 32;
  Resources.maxFragmentUniformComponents = 4096;
  Resources.maxDrawBuffers = 32;
  Resources.maxVertexUniformVectors = 128;
  Resources.maxVaryingVectors = 8;
  Resources.maxFragmentUniformVectors = 16;
  Resources.maxVertexOutputVectors = 16;
  Resources.maxFragmentInputVectors = 15;
  Resources.minProgramTexelOffset = -8;
  Resources.maxProgramTexelOffset = 7;
  Resources.maxClipDistances = 8;
  Resources.maxComputeWorkGroupCountX = 65535;
  Resources.maxComputeWorkGroupCountY = 65535;
  Resources.maxComputeWorkGroupCountZ = 65535;
  Resources.maxComputeWorkGroupSizeX = 1024;
  Resources.maxComputeWorkGroupSizeY = 1024;
  Resources.maxComputeWorkGroupSizeZ = 64;
  Resources.maxComputeUniformComponents = 1024;
  Resources.maxComputeTextureImageUnits = 16;
  Resources.maxComputeImageUniforms = 8;
  Resources.maxComputeAtomicCounters = 8;
  Resources.maxComputeAtomicCounterBuffers = 1;
  Resources.maxVaryingComponents = 60;
  Resources.maxVertexOutputComponents = 64;
  Resources.maxGeometryInputComponents = 64;
  Resources.maxGeometryOutputComponents = 128;
  Resources.maxFragmentInputComponents = 128;
  Resources.maxImageUnits = 8;
  Resources.maxCombinedImageUnitsAndFragmentOutputs = 8;
  Resources.maxCombinedShaderOutputResources = 8;
  Resources.maxImageSamples = 0;
  Resources.maxVertexImageUniforms = 0;
  Resources.maxTessControlImageUniforms = 0;
  Resources.maxTessEvaluationImageUniforms = 0;
  Resources.maxGeometryImageUniforms = 0;
  Resources.maxFragmentImageUniforms = 8;
  Resources.maxCombinedImageUniforms = 8;
  Resources.maxGeometryTextureImageUnits = 16;
  Resources.maxGeometryOutputVertices = 256;
  Resources.maxGeometryTotalOutputComponents = 1024;
  Resources.maxGeometryUniformComponents = 1024;
  Resources.maxGeometryVaryingComponents = 64;
  Resources.maxTessControlInputComponents = 128;
  Resources.maxTessControlOutputComponents = 128;
  Resources.maxTessControlTextureImageUnits = 16;
  Resources.maxTessControlUniformComponents = 1024;
  Resources.maxTessControlTotalOutputComponents = 4096;
  Resources.maxTessEvaluationInputComponents = 128;
  Resources.maxTessEvaluationOutputComponents = 128;
  Resources.maxTessEvaluationTextureImageUnits = 16;
  Resources.maxTessEvaluationUniformComponents = 1024;
  Resources.maxTessPatchComponents = 120;
  Resources.maxPatchVertices = 32;
  Resources.maxTessGenLevel = 64;
  Resources.maxViewports = 16;
  Resources.maxVertexAtomicCounters = 0;
  Resources.maxTessControlAtomicCounters = 0;
  Resources.maxTessEvaluationAtomicCounters = 0;
  Resources.maxGeometryAtomicCounters = 0;
  Resources.maxFragmentAtomicCounters = 8;
  Resources.maxCombinedAtomicCounters = 8;
  Resources.maxAtomicCounterBindings = 1;
  Resources.maxVertexAtomicCounterBuffers = 0;
  Resources.maxTessControlAtomicCounterBuffers = 0;
  Resources.maxTessEvaluationAtomicCounterBuffers = 0;
  Resources.maxGeometryAtomicCounterBuffers = 0;
  Resources.maxFragmentAtomicCounterBuffers = 1;
  Resources.maxCombinedAtomicCounterBuffers = 1;
  Resources.maxAtomicCounterBufferSize = 16384;
  Resources.maxTransformFeedbackBuffers = 4;
  Resources.maxTransformFeedbackInterleavedComponents = 64;
  Resources.maxCullDistances = 8;
  Resources.maxCombinedClipAndCullDistances = 8;
  Resources.maxSamples = 4;
  Resources.maxMeshOutputVerticesNV = 256;
  Resources.maxMeshOutputPrimitivesNV = 512;
  Resources.maxMeshWorkGroupSizeX_NV = 32;
  Resources.maxMeshWorkGroupSizeY_NV = 1;
  Resources.maxMeshWorkGroupSizeZ_NV = 1;
  Resources.maxTaskWorkGroupSizeX_NV = 32;
  Resources.maxTaskWorkGroupSizeY_NV = 1;
  Resources.maxTaskWorkGroupSizeZ_NV = 1;
  Resources.maxMeshViewCountNV = 4;
  Resources.limits.nonInductiveForLoops = 1;
  Resources.limits.whileLoops = 1;
  Resources.limits.doWhileLoops = 1;
  Resources.limits.generalUniformIndexing = 1;
  Resources.limits.generalAttributeMatrixVectorIndexing = 1;
  Resources.limits.generalVaryingIndexing = 1;
  Resources.limits.generalSamplerIndexing = 1;
  Resources.limits.generalVariableIndexing = 1;
  Resources.limits.generalConstantMatrixVectorIndexing = 1;
}

//
// Compile a given string containing GLSL into SPV for use by VK
// Return value of false means an error was encountered.
//
bool GLSLtoSPV(const VkShaderStageFlagBits shader_type, const char *pshader, std::vector<unsigned int> &spirv)
{
  EShLanguage stage = FindLanguage(shader_type);
  glslang::TShader shader(stage);
  glslang::TProgram program;
  const char *shaderStrings[1];
  TBuiltInResource Resources = {};
  init_resources(Resources);

  // Enable SPIR-V and Vulkan rules when parsing GLSL
  EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

  shaderStrings[0] = pshader;
  shader.setStrings(shaderStrings, 1);

  if(!shader.parse(&Resources, 100, false, messages)) {
    puts(shader.getInfoLog());
    puts(shader.getInfoDebugLog());
    return false;  // something didn't work
  }

  program.addShader(&shader);

  // Program-level processing...
  if(!program.link(messages)) {
    puts(shader.getInfoLog());
    puts(shader.getInfoDebugLog());
    fflush(stdout);
    return false;
  }

  glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);

  return true;
}

int setupShaders()
{
  static const char *vertShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (std140, binding = 0) uniform bufferVals {\n"
    "    mat4 mvp;\n"
    "} myBufferVals;\n"
    "layout (location = 0) in vec4 pos;\n"
    "layout (location = 1) in vec4 inColor;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main() {\n"
    "   outColor = inColor;\n"
    "   gl_Position = myBufferVals.mvp * pos;\n"
    "}\n";

  static const char *fragShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (location = 0) in vec4 color;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main() {\n"
    "   outColor = color;\n"
    "}\n";

  glslang::InitializeProcess();

  std::vector<unsigned int> vtx_spv;
  auto res = GLSLtoSPV(VK_SHADER_STAGE_VERTEX_BIT, vertShaderText, vtx_spv);
  assert(res);

  std::vector<unsigned int> frag_spv;
  res = GLSLtoSPV(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderText, frag_spv);
  assert(res);

  glslang::FinalizeProcess();

  // Vertex shader
  const auto vertexShaderModuleCreationInfo = vk::ShaderModuleCreateInfo()
    .setCodeSize(vtx_spv.size() * sizeof(vtx_spv.front()))
    .setPCode(vtx_spv.data());
  const auto vertexShaderModule = info.device.createShaderModule(vertexShaderModuleCreationInfo);

  // Fragment shader
  const auto fragmentShaderModuleCreationInfo = vk::ShaderModuleCreateInfo()
    .setCodeSize(frag_spv.size() * sizeof(frag_spv.front()))
    .setPCode(frag_spv.data());
  const auto fragmentShaderModule = info.device.createShaderModule(fragmentShaderModuleCreationInfo);

  const auto pipelineStageShaderStageCreateInfo = std::vector<vk::PipelineShaderStageCreateInfo>
  {
    // Vertex
    vk::PipelineShaderStageCreateInfo()
      .setStage(vk::ShaderStageFlagBits::eVertex)
      .setPName("main")
      .setModule(vertexShaderModule)
    ,
    vk::PipelineShaderStageCreateInfo()
      .setStage(vk::ShaderStageFlagBits::eFragment)
      .setPName("main")
      .setModule(fragmentShaderModule)
  };

  return 0;
}

int main()
{
#ifdef WIN32
  std::cout.rdbuf(&g_DebugStreamFor_cout); // Redirect std::cout to OutputDebugString!
#endif

  setupApplicationAndInstance();
  setupDevicesAndQueues();
  setupSwapChains();
  setupDepth();

  const auto proj = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
  const auto view = glm::lookAt(glm::vec3(-5, 3, -10), // Camera is at (-5,3,-10), in World Space
                                glm::vec3(0, 0, 0),    // and looks at the origin
                                glm::vec3(0, -1, 0));  // Head is up (set to 0,-1,0 to look upside-down)
  const auto model = glm::mat4(1.0f);
  // Vulkan clip space has inverted Y and half Z.
  const auto clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, -1.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.5f, 0.0f,
                              0.0f, 0.0f, 0.5f, 1.0f);
  const auto mvp = clip * proj * view * model;

  setupUniformBuffer((void*)&mvp, sizeof(mvp));

  setupDescriptorSetLayoutAndPipelineLayout();
  setupDescriptorSetPool();
  allocateDescriptorSet();
  initRenderPass();

  // Poll for user input.
  bool stillRunning = true;
  while(stillRunning) {

    SDL_Event event;
    while(SDL_PollEvent(&event)) {

      switch(event.type) {

      case SDL_QUIT:
        stillRunning = false;
        break;

      default:
        // Do nothing.
        break;
      }
    }

    SDL_Delay(10);
  }

  // Clean up.
  info.inst.destroySurfaceKHR(info.surface);
  info.inst.destroy();
  SDL_DestroyWindow(sdlinfo.window);
  SDL_Quit();

  return 0;
}

vk::SurfaceKHR createVulkanSurface(const vk::Instance& instance, SDL_Window* window)
{
  SDL_SysWMinfo windowInfo;
  SDL_VERSION(&windowInfo.version);
  if(!SDL_GetWindowWMInfo(window, &windowInfo)) {
    throw std::system_error(std::error_code(), "SDK window manager info is not available.");
  }

  switch(windowInfo.subsystem) {

#if defined(SDL_VIDEO_DRIVER_ANDROID) && defined(VK_USE_PLATFORM_ANDROID_KHR)
  case SDL_SYSWM_ANDROID:
  {
    vk::AndroidSurfaceCreateInfoKHR surfaceInfo = vk::AndroidSurfaceCreateInfoKHR()
      .setWindow(windowInfo.info.android.window);
    return instance.createAndroidSurfaceKHR(surfaceInfo);
  }
#endif

#if defined(SDL_VIDEO_DRIVER_WAYLAND) && defined(VK_USE_PLATFORM_WAYLAND_KHR)
  case SDL_SYSWM_WAYLAND:
  {
    vk::WaylandSurfaceCreateInfoKHR surfaceInfo = vk::WaylandSurfaceCreateInfoKHR()
      .setDisplay(windowInfo.info.wl.display)
      .setSurface(windowInfo.info.wl.surface);
    return instance.createWaylandSurfaceKHR(surfaceInfo);
  }
#endif

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && defined(VK_USE_PLATFORM_WIN32_KHR)
  case SDL_SYSWM_WINDOWS:
  {
    vk::Win32SurfaceCreateInfoKHR surfaceInfo = vk::Win32SurfaceCreateInfoKHR()
      .setHinstance(GetModuleHandle(NULL))
      .setHwnd(windowInfo.info.win.window);
    return instance.createWin32SurfaceKHR(surfaceInfo);
  }
#endif

#if defined(SDL_VIDEO_DRIVER_X11) && defined(VK_USE_PLATFORM_XLIB_KHR)
  case SDL_SYSWM_X11:
  {
    vk::XlibSurfaceCreateInfoKHR surfaceInfo = vk::XlibSurfaceCreateInfoKHR()
      .setDpy(windowInfo.info.x11.display)
      .setWindow(windowInfo.info.x11.window);
    return instance.createXlibSurfaceKHR(surfaceInfo);
  }
#endif

  default:
    throw std::system_error(std::error_code(), "Unsupported window manager is in use.");
  }
}

std::vector<const char*> getAvailableWSIExtensions()
{
  std::vector<const char*> extensions;
  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
  extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WIN32_KHR)
  extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
  extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif

  return extensions;
}
