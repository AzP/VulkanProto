/*
 * Vulkan C++ Windowed Project Template
 * Create and destroy a Vulkan surface on an SDL window.
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

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <vulkan/vulkan.hpp>

#undef min
#undef max

#ifdef WIN32
// Class for redirecting cout to Visual Studio output window
class dbg_stream_for_cout : public std::stringbuf {
public:
	~dbg_stream_for_cout() { sync(); }
	int sync() {
		::OutputDebugStringA(str().c_str());
		str(std::string()); // Clear the string buffer
		return 0;
	}
};
dbg_stream_for_cout g_DebugStreamFor_cout;
#endif

struct DepthData {
	vk::Image image;
	vk::DeviceMemory mem;
	vk::ImageView view;
	vk::Format format;
};

struct SwapchainBuffers {
	vk::Image image;
	vk::CommandBuffer cmd;
	vk::CommandBuffer graphics_to_present_cmd;
	vk::ImageView view;
};

struct UniformData {
	vk::Buffer buffer;
	vk::DeviceMemory mem;
	vk::DescriptorBufferInfo bufferInfo;
};

struct VertexBufferData {
	vk::Buffer buffer;
	vk::DeviceMemory mem;
	vk::VertexInputBindingDescription vertexInputDesc;
	std::vector<vk::VertexInputAttributeDescription> vertexInputAttribs;
};

struct vkinfo {
	std::vector<vk::PhysicalDevice> gpus;
	vk::PhysicalDevice gpu;
	vk::Device device;
	vk::SurfaceKHR surface;
	vk::SurfaceCapabilitiesKHR surfaceCapabilities;
	vk::Instance inst;

	vk::CommandPool cmdPool;
	std::vector<vk::CommandBuffer> cmdBuffers;

	uint32_t gfxQueueFamilyIdx{0}, prsntQueueFamilyIdx{0};
	vk::Queue gfxQueue;
	vk::Queue prsntQueue;

	vk::SwapchainKHR swapchain;
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

	std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;

	std::vector<vk::Framebuffer> framebuffers;

	VertexBufferData vertexBufferData;

	vk::PipelineCache pipelineCache;
	vk::Pipeline pipeline;
	uint32_t currentBuffer;
} info;

struct sdlinfo {
	SDL_Window *window;
} sdlinfo;

vk::PresentModeKHR getPresentMode(const vk::SurfaceKHR &surface, const vk::PhysicalDevice &gpu) {
	auto modes = gpu.getSurfacePresentModesKHR(surface);
	if (modes.size() == 1)
		return modes.at(0);

	for (const auto &mode : modes)
		if (mode == vk::PresentModeKHR::eMailbox)
			return mode;

	return vk::PresentModeKHR::eFifo;
}

vk::SurfaceKHR createVulkanSurface(const vk::Instance &instance, SDL_Window *window);
std::vector<const char *> getAvailableWSIExtensions();

bool memory_type_from_properties(uint32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t &typeIndex) {
	// Search memtypes to find first index with those properties
	for (uint32_t i = 0; i < info.memoryProperties.memoryTypeCount; ++i) {
		if ((typeBits & 1) == 1) {
			// Type is available, does it match user properties?
			if ((info.memoryProperties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
				typeIndex = i;
				return true;
			}
		}
		typeBits >>= 1;
	}
	// No memory types matched, return failure
	return false;
}

int setupApplicationAndInstance() {
	// Use validation layers if this is a debug build, and use WSI extensions
	// regardless
	std::vector<const char *> extensions = getAvailableWSIExtensions();
	std::vector<const char *> layers;
	//#if defined(_DEBUG)
	layers.push_back("VK_LAYER_KHRONOS_validation");
	//#endif

	// vk::ApplicationInfo allows the programmer to specifiy some basic
	// information about the program, which can be useful for layers and tools to
	// provide more debug information.
	vk::ApplicationInfo appInfo = vk::ApplicationInfo()
		.setPApplicationName("Peppes ultimata Vulkan-program")
		.setApplicationVersion(1)
		.setPEngineName("Peppes Motor")
		.setEngineVersion(1)
		.setApiVersion(VK_API_VERSION_1_0);

	// vk::InstanceCreateInfo is where the programmer specifies the layers and/or
	// extensions that are needed.
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
	} catch (const std::exception &e) {
		std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
		return 1;
	}
	info.inst = instance;

	// Create an SDL window that supports Vulkan and OpenGL rendering.
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::cout << "Could not initialize SDL." << std::endl;
		return 1;
	}
	SDL_Window *window = SDL_CreateWindow("Vulkan Window", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 720, 720, SDL_WINDOW_OPENGL);
	if (window == NULL) {
		std::cout << "Could not create SDL window." << std::endl;
		return 1;
	}
	sdlinfo.window = window;

	// Create a platform specific Vulkan surface for rendering
	vk::SurfaceKHR surface;
	try {
		surface = createVulkanSurface(instance, window);
	} catch (const std::exception &e) {
		std::cout << "Failed to create Vulkan surface: " << e.what() << std::endl;
		instance.destroy();
		return 1;
	}
	info.surface = surface;

	return 0;
}

int setupDevicesAndQueues() {
	// Enumerate devices
	try {
		info.gpus = info.inst.enumeratePhysicalDevices();
		info.gpu = info.gpus[0];
	} catch (const std::exception &e) {
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
	for (uint32_t i = 0; i < queueFamilyProps.size(); ++i) {
		info.gpu.getSurfaceSupportKHR(i, info.surface, &supportsPresent[i]);
	}

	uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
	uint32_t presentQueueFamilyIndex = UINT32_MAX;
	for (uint32_t i = 0; i < queueFamilyProps.size(); ++i) {
		if (queueFamilyProps[i].queueFlags & vk::QueueFlagBits::eGraphics) {
			if (graphicsQueueFamilyIndex == UINT32_MAX) {
				graphicsQueueFamilyIndex = i;
			}

			if (supportsPresent[i] == VK_TRUE) {
				graphicsQueueFamilyIndex = i;
				presentQueueFamilyIndex = i;
				break;
			}
		}
	}

	if (presentQueueFamilyIndex == UINT32_MAX) {
		// If didn't find a queue that supports both graphics and present,
		// then find a separate present queue.
		for (uint32_t i = 0; i < queueFamilyProps.size(); ++i) {
			if (supportsPresent[i] == VK_TRUE) {
				presentQueueFamilyIndex = i;
				break;
			}
		}
	}

	// Generate error if could not find both a graphics and a present queue
	if (graphicsQueueFamilyIndex == UINT32_MAX || presentQueueFamilyIndex == UINT32_MAX) {
		std::cout << "Could not find both graphics and present queues" << std::endl;
		info.inst.destroy();
		return 1;
	}

	info.gfxQueueFamilyIdx = graphicsQueueFamilyIndex;
	info.prsntQueueFamilyIdx = presentQueueFamilyIndex;

	// Setup queue creation info
	float queuePriorities[1] = {0.0f};
	auto queueInfo = vk::DeviceQueueCreateInfo().setQueueCount(1).setPQueuePriorities(queuePriorities).setQueueFamilyIndex(graphicsQueueFamilyIndex);

	const std::vector<const char *> extensionNames = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	const auto createInfo = vk::DeviceCreateInfo()
		.setPQueueCreateInfos(&queueInfo)
		.setQueueCreateInfoCount(1)
		.setEnabledExtensionCount(1)
		.setPpEnabledExtensionNames(extensionNames.data());

	try {
		info.device = info.gpus[0].createDevice(createInfo);
	} catch (const std::exception &e) {
		std::cout << "Unable to create vkDevice: " << e.what() << std::endl;
		info.inst.destroy();
		return 1;
	}

	// Save handles to the Queues
	info.gfxQueue = info.device.getQueue(graphicsQueueFamilyIndex, 0);
	info.prsntQueue = info.device.getQueue(presentQueueFamilyIndex, 0);

	return 0;
}

int setupCommandBuffers() {
	// Setup and create a command queue pool
	const auto cmdPoolInfo = vk::CommandPoolCreateInfo().setQueueFamilyIndex(info.gfxQueueFamilyIdx);
	info.cmdPool = info.device.createCommandPool(cmdPoolInfo);
	// Allocate the command buffers for that pool
	const auto cmdBufAllocInfo = vk::CommandBufferAllocateInfo().setCommandBufferCount(1).setCommandPool(info.cmdPool);
	info.cmdBuffers = info.device.allocateCommandBuffers(cmdBufAllocInfo);

	return 0;
}

void beginCommandBuffer() {
	const auto cmdBufferInfo = vk::CommandBufferBeginInfo();
	info.cmdBuffers.front().begin(cmdBufferInfo);
}

void endCommandBuffer() { info.cmdBuffers.front().end(); }

int setupSwapChains() {
	// Create Swapchains for our platform specific surface
	const auto oldSwapChain = info.swapchain;
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
	if (info.gfxQueue != info.prsntQueue) {
		std::vector<uint32_t> indices = {info.gfxQueueFamilyIdx, info.prsntQueueFamilyIdx};
		swapchainCreateInfo.setImageSharingMode(vk::SharingMode::eConcurrent)
			.setQueueFamilyIndexCount((uint32_t)indices.size())
			.setPQueueFamilyIndices(indices.data());
	}
	info.surfaceCapabilities = pSurfCap;

	// Create the swapchain, clear old one, and create images and imageviews
	info.swapchain = info.device.createSwapchainKHR(swapchainCreateInfo);
	if (oldSwapChain)
		info.device.destroySwapchainKHR(oldSwapChain);

	auto swapChainImages = info.device.getSwapchainImagesKHR(info.swapchain);
	info.swapchainBuffers.resize(swapChainImages.size());
	auto imageViewCreateInfo = vk::ImageViewCreateInfo()
		.setViewType(vk::ImageViewType::e2D)
		.setFormat(formats.at(0).format)
		.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

	// Initialize the images using the same info
	for (uint32_t i = 0u; i < swapChainImages.size(); ++i) {
		imageViewCreateInfo.setImage(swapChainImages[i]);
		info.swapchainBuffers[i].image = swapChainImages[i];
		info.swapchainBuffers[i].view = info.device.createImageView(imageViewCreateInfo);
	}

	return 0;
}

int setupDepth() {
	// Setup depth buffer
	info.depth.format = vk::Format::eD16Unorm;
	const auto depthImageInfo = vk::ImageCreateInfo()
		.setFormat(info.depth.format)
		.setImageType(vk::ImageType::e2D)
		.setExtent({info.surfaceCapabilities.currentExtent.width, info.surfaceCapabilities.currentExtent.height, 1})
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(vk::SampleCountFlagBits::e1)
		.setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment);
	info.depth.image = info.device.createImage(depthImageInfo);

	// We need memory requirements from the GPU to allocate depth buffer memory
	auto memReqs = info.device.getImageMemoryRequirements(info.depth.image);
	auto memAllocInfo = vk::MemoryAllocateInfo().setAllocationSize(memReqs.size).setMemoryTypeIndex(0);
	const auto pass = memory_type_from_properties(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, memAllocInfo.memoryTypeIndex);
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

int setupUniformBuffer(void *data, size_t size) {
	const auto buf_info = vk::BufferCreateInfo()
		.setUsage(vk::BufferUsageFlagBits::eUniformBuffer).setSize(size);
	const auto buffer = info.device.createBuffer(buf_info); // Create buffer object

	// We need memory requirements from the GPU to allocate uniform buffer memory
	auto memReqs = info.device.getBufferMemoryRequirements(buffer);
	auto memAllocInfo = vk::MemoryAllocateInfo().setAllocationSize(memReqs.size).setMemoryTypeIndex(0);
	const auto pass = memory_type_from_properties(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			memAllocInfo.memoryTypeIndex);
	assert(pass && "No mappable, coherent memory");
	const auto memory = info.device.allocateMemory(memAllocInfo);

	// Write the actual uniform data to the buffer
	const auto buf_data = info.device.mapMemory(memory, 0, memReqs.size);
	memcpy(buf_data, data, size);
	info.device.unmapMemory(memory); // Unmap the memory buffer ASAP
	info.device.bindBufferMemory(buffer, memory,
			0); // Associate the allocated memory with the buffer object

	// Save buffer info for later access
	info.uniformData.buffer = buffer;
	info.uniformData.mem = memory;
	info.uniformData.bufferInfo.buffer = info.uniformData.buffer;
	info.uniformData.bufferInfo.offset = 0;
	info.uniformData.bufferInfo.range = size;

	return 0;
}

int setupDescriptorSetLayoutAndPipelineLayout() {
	// Number of descriptor sets
	const auto numOfDescriptors = 1;
	// Create a layout binding for a uniform buffer used in vertex shader
	const auto layoutBinding = vk::DescriptorSetLayoutBinding()
		.setDescriptorType(vk::DescriptorType::eUniformBuffer)
		.setDescriptorCount(numOfDescriptors)
		.setStageFlags(vk::ShaderStageFlagBits::eVertex);
	const auto descriptorLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo().setBindingCount(1).setPBindings(&layoutBinding);
	// Create descriptorLayout and save it in info object
	const auto descriptorSetLayout = info.device.createDescriptorSetLayout(descriptorLayoutCreateInfo);
	info.descriptorSetLayouts.push_back(descriptorSetLayout);

	const auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo().setSetLayoutCount(numOfDescriptors).setPSetLayouts(&descriptorSetLayout);
	info.pipelineLayout = info.device.createPipelineLayout(pipelineLayoutCreateInfo);

	/*
	   In GLSL:
	   layout (set=M, binding=N) uniform sampler2D variableNameArray[I];

	   M refers the the M'th descriptor set layout in the pSetLayouts member of the
	   pipeline layout N refers to the N'th descriptor set (binding) in M's pBindings
	   member of the descriptor set layout I is the index into the array of
	   descriptors in N's descriptor set
	   */

	return 0;
}

int setupDescriptorSetPool() {
	const auto typeCount = vk::DescriptorPoolSize().setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(1);
	const auto descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo().setMaxSets(1).setPoolSizeCount(1).setPPoolSizes(&typeCount);
	info.descriptorPool = info.device.createDescriptorPool(descriptorPoolCreateInfo);

	return 0;
}

int allocateDescriptorSet() {
	const auto allocInfo = vk::DescriptorSetAllocateInfo()
		.setDescriptorPool(info.descriptorPool)
		.setDescriptorSetCount((uint32_t)info.descriptorSetLayouts.size())
		.setPSetLayouts(info.descriptorSetLayouts.data());
	info.descriptorSets = info.device.allocateDescriptorSets(allocInfo);

	const auto writes = std::vector<vk::WriteDescriptorSet>{vk::WriteDescriptorSet()
		.setDstSet(info.descriptorSets.front())
			.setDescriptorCount((uint32_t)info.descriptorSets.size())
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setPBufferInfo((const vk::DescriptorBufferInfo *)&info.uniformData.bufferInfo)};
	// Copy the VkDescriptorBufferInfo into the descriptor (the device)
	info.device.updateDescriptorSets(writes, nullptr);

	return 0;
}

int initRenderPass() {
	// Set up two attachments
	const auto attachments = std::vector<vk::AttachmentDescription>{// The framebuffer (color) attachment
		vk::AttachmentDescription()
			.setFormat(info.surfaceFormat)
			.setSamples(vk::SampleCountFlagBits::e1)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
			.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
			.setInitialLayout(vk::ImageLayout::eUndefined)
			.setFinalLayout(vk::ImageLayout::ePresentSrcKHR),
			// The depth attachment
			vk::AttachmentDescription()
				.setFormat(info.depth.format)
				.setSamples(vk::SampleCountFlagBits::e1)
				.setLoadOp(vk::AttachmentLoadOp::eClear)
				.setStoreOp(vk::AttachmentStoreOp::eDontCare)
				.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
				.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
				.setInitialLayout(vk::ImageLayout::eUndefined)
				.setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)};

	// AttachmentReferences
	const auto colorReference = vk::AttachmentReference()
		.setAttachment(0) // Matches the array index above
		.setLayout(vk::ImageLayout::eColorAttachmentOptimal);
	const auto depthReference = vk::AttachmentReference().setAttachment(1).setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

	// Subpass description
	const auto subpass = vk::SubpassDescription()
		.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics) // Graphics or compute
		.setColorAttachmentCount(1)
		.setPColorAttachments(&colorReference)
		.setPDepthStencilAttachment(&depthReference);

	// Define the render pass
	const auto renderPassCreateInfo = vk::RenderPassCreateInfo()
		.setAttachmentCount((uint32_t)attachments.size())
		.setPAttachments(attachments.data())
		.setSubpassCount(1)
		.setPSubpasses(&subpass);
	info.renderPass = info.device.createRenderPass(renderPassCreateInfo);

	return 0;
}

// Read spv compiled binary shader and return it as a vector<unsigned int>
std::vector<unsigned int> readShader(std::string filename) {
	std::ifstream shaderfile(filename, std::ifstream::binary);
	assert(shaderfile);
	std::vector<unsigned int> contents;
	shaderfile.seekg(0, std::ios::end);
	contents.resize(shaderfile.tellg() / sizeof(contents.front()));
	shaderfile.seekg(0, std::ios::beg);
	shaderfile.read((char *)contents.data(), contents.size() * sizeof(contents.front()));
	shaderfile.close();

	return contents;
}

int setupShaders() {
	const auto vtx_spv = readShader("vert.spv");
	const auto frag_spv = readShader("frag.spv");

	// Vertex shader
	const auto vertexShaderModuleCreationInfo = vk::ShaderModuleCreateInfo().setCodeSize(vtx_spv.size() * sizeof(vtx_spv.front())).setPCode(vtx_spv.data());
	const auto vertexShaderModule = info.device.createShaderModule(vertexShaderModuleCreationInfo);

	// Fragment shader
	const auto fragmentShaderModuleCreationInfo = vk::ShaderModuleCreateInfo().setCodeSize(frag_spv.size() * sizeof(frag_spv.front())).setPCode(frag_spv.data());
	const auto fragmentShaderModule = info.device.createShaderModule(fragmentShaderModuleCreationInfo);

	const auto pipelineStageShaderStageCreateInfo = std::vector<vk::PipelineShaderStageCreateInfo>{
		// Vertex
		vk::PipelineShaderStageCreateInfo().setStage(vk::ShaderStageFlagBits::eVertex).setPName("main").setModule(vertexShaderModule),
			vk::PipelineShaderStageCreateInfo().setStage(vk::ShaderStageFlagBits::eFragment).setPName("main").setModule(fragmentShaderModule)};

	// Save them for later when setting up the graphics pipeline
	info.shaderStages = pipelineStageShaderStageCreateInfo;

	return 0;
}

int setupFrameBuffers() {
	auto attachments = std::vector<vk::ImageView>(2);
	attachments[1] = info.depth.view; // Set the second buffer to always be depth buffer

	const auto frameBufferCreateInfo = vk::FramebufferCreateInfo()
		.setRenderPass(info.renderPass)
		.setAttachmentCount(2)
		.setPAttachments(attachments.data())
		.setWidth(info.surfaceCapabilities.currentExtent.width)
		.setHeight(info.surfaceCapabilities.currentExtent.height)
		.setLayers(1);

	info.framebuffers = std::vector<vk::Framebuffer>(info.swapchainBuffers.size());

	for (unsigned int i = 0; i < info.framebuffers.size(); ++i) {
		// Set the first (fb) attachment to the buffer and then create the
		// framebuffer (using the attachments)
		attachments[0] = info.swapchainBuffers[i].view;
		info.framebuffers[i] = info.device.createFramebuffer(frameBufferCreateInfo);
	}

	return 0;
}

int createVertexBuffer() {
	const auto vertexData = std::vector<glm::vec4>{
		// red face
		glm::vec4(-1, -1, 1, 1),
			glm::vec4(1.f, 0.f, 0.f, 1.0f),
			glm::vec4(-1, 1, 1, 1),
			glm::vec4(1.f, 0.f, 0.f, 1.0f),
			glm::vec4(1, -1, 1, 1),
			glm::vec4(1.f, 0.f, 0.f, 1.0f),
			glm::vec4(1, -1, 1, 1),
			glm::vec4(1.f, 0.f, 0.f, 1.0f),
			glm::vec4(-1, 1, 1, 1),
			glm::vec4(1.f, 0.f, 0.f, 1.0f),
			glm::vec4(1, 1, 1, 1),
			glm::vec4(1.f, 0.f, 0.f, 1.0f),
			// green face
			glm::vec4(-1, -1, -1, 1),
			glm::vec4(0.f, 1.f, 0.f, 1.0f),
			glm::vec4(1, -1, -1, 1),
			glm::vec4(0.f, 1.f, 0.f, 1.0f),
			glm::vec4(-1, 1, -1, 1),
			glm::vec4(0.f, 1.f, 0.f, 1.0f),
			glm::vec4(-1, 1, -1, 1),
			glm::vec4(0.f, 1.f, 0.f, 1.0f),
			glm::vec4(1, -1, -1, 1),
			glm::vec4(0.f, 1.f, 0.f, 1.0f),
			glm::vec4(1, 1, -1, 1),
			glm::vec4(0.f, 1.f, 0.f, 1.0f),
			// blue face
			glm::vec4(-1, 1, 1, 1),
			glm::vec4(0.f, 0.f, 1.f, 1.0f),
			glm::vec4(-1, -1, 1, 1),
			glm::vec4(0.f, 0.f, 1.f, 1.0f),
			glm::vec4(-1, 1, -1, 1),
			glm::vec4(0.f, 0.f, 1.f, 1.0f),
			glm::vec4(-1, 1, -1, 1),
			glm::vec4(0.f, 0.f, 1.f, 1.0f),
			glm::vec4(-1, -1, 1, 1),
			glm::vec4(0.f, 0.f, 1.f, 1.0f),
			glm::vec4(-1, -1, -1, 1),
			glm::vec4(0.f, 0.f, 1.f, 1.0f),
			// yellow face
			glm::vec4(1, 1, 1, 1),
			glm::vec4(1.f, 1.f, 0.f, 1.0f),
			glm::vec4(1, 1, -1, 1),
			glm::vec4(1.f, 1.f, 0.f, 1.0f),
			glm::vec4(1, -1, 1, 1),
			glm::vec4(1.f, 1.f, 0.f, 1.0f),
			glm::vec4(1, -1, 1, 1),
			glm::vec4(1.f, 1.f, 0.f, 1.0f),
			glm::vec4(1, 1, -1, 1),
			glm::vec4(1.f, 1.f, 0.f, 1.0f),
			glm::vec4(1, -1, -1, 1),
			glm::vec4(1.f, 1.f, 0.f, 1.0f),
			// magenta face
			glm::vec4(1, 1, 1, 1),
			glm::vec4(1.f, 0.f, 1.f, 1.0f),
			glm::vec4(-1, 1, 1, 1),
			glm::vec4(1.f, 0.f, 1.f, 1.0f),
			glm::vec4(1, 1, -1, 1),
			glm::vec4(1.f, 0.f, 1.f, 1.0f),
			glm::vec4(1, 1, -1, 1),
			glm::vec4(1.f, 0.f, 1.f, 1.0f),
			glm::vec4(-1, 1, 1, 1),
			glm::vec4(1.f, 0.f, 1.f, 1.0f),
			glm::vec4(-1, 1, -1, 1),
			glm::vec4(1.f, 0.f, 1.f, 1.0f),
			// cyan face
			glm::vec4(1, -1, 1, 1),
			glm::vec4(0.f, 1.f, 1.f, 1.0f),
			glm::vec4(1, -1, -1, 1),
			glm::vec4(0.f, 1.f, 1.f, 1.0f),
			glm::vec4(-1, -1, 1, 1),
			glm::vec4(0.f, 1.f, 1.f, 1.0f),
			glm::vec4(-1, -1, 1, 1),
			glm::vec4(0.f, 1.f, 1.f, 1.0f),
			glm::vec4(1, -1, -1, 1),
			glm::vec4(0.f, 1.f, 1.f, 1.0f),
			glm::vec4(-1, -1, -1, 1),
			glm::vec4(0.f, 1.f, 1.f, 1.0f),
	};

	// Create the buffer create info description, and then create the buffer
	const auto bufferInfo = vk::BufferCreateInfo()
		.setUsage(vk::BufferUsageFlagBits::eVertexBuffer)
		.setSharingMode(vk::SharingMode::eExclusive)
		.setSize(vertexData.size() * sizeof(vertexData.front()));
	const auto buffer = info.device.createBuffer(bufferInfo);

	// We need memory requirements from the GPU to allocate vertex buffer memory
	auto memReqs = info.device.getBufferMemoryRequirements(buffer);
	auto memAllocInfo = vk::MemoryAllocateInfo().setAllocationSize(memReqs.size).setMemoryTypeIndex(0);
	const auto pass = memory_type_from_properties(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			memAllocInfo.memoryTypeIndex);
	assert(pass && "No mappable, coherent memory");
	const auto memory = info.device.allocateMemory(memAllocInfo);

	// Write the actual vertex data to the buffer
	const auto bufferData = info.device.mapMemory(memory, 0, memReqs.size);
	memcpy(bufferData, vertexData.data(), sizeof(vertexData.front()) * vertexData.size());

	info.device.unmapMemory(memory); // Unmap the memory buffer ASAP
	info.device.bindBufferMemory(buffer, memory,
			0); // Associate the allocated memory with the buffer object
	info.vertexBufferData.buffer = buffer;
	info.vertexBufferData.mem = memory;

	// Save the vertex info for later setup of graphics pipeline
	info.vertexBufferData.vertexInputDesc =
		vk::VertexInputBindingDescription().setBinding(0).setInputRate(vk::VertexInputRate::eVertex).setStride((uint32_t)(sizeof(vertexData.front()) * 2.0f));

	info.vertexBufferData.vertexInputAttribs = std::vector<vk::VertexInputAttributeDescription>{
		vk::VertexInputAttributeDescription().setBinding(0).setLocation(0).setFormat(vk::Format::eR32G32B32A32Sfloat).setOffset(0),
			vk::VertexInputAttributeDescription().setBinding(0).setLocation(1).setFormat(vk::Format::eR32G32B32A32Sfloat).setOffset(sizeof(vertexData.front()))};

	return 0;
}

int setupPipelineStates() {
	// Pipeline Vertex Input State
	const auto vertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo()
		.setVertexBindingDescriptionCount(1)
		.setPVertexBindingDescriptions(&info.vertexBufferData.vertexInputDesc)
		.setVertexAttributeDescriptionCount(info.vertexBufferData.vertexInputAttribs.size())
		.setPVertexAttributeDescriptions(info.vertexBufferData.vertexInputAttribs.data());

	// Pipeline Vertex Input Assembly State (describes the mesh data)
	const auto vertexInputAssemblyStateCreateInfo =
		vk::PipelineInputAssemblyStateCreateInfo().setPrimitiveRestartEnable(VK_FALSE).setTopology(vk::PrimitiveTopology::eTriangleList);

	// Pipeline rasterization state
	const auto pipelineRasterStateCreateInfo = vk::PipelineRasterizationStateCreateInfo()
		.setPolygonMode(vk::PolygonMode::eFill)
		.setCullMode(vk::CullModeFlagBits::eBack)
		.setFrontFace(vk::FrontFace::eClockwise)
		.setDepthClampEnable(VK_FALSE)
		.setLineWidth(1.0);

	// Pipeline Color Blend State
	const auto attachState = std::vector<vk::PipelineColorBlendAttachmentState>{vk::PipelineColorBlendAttachmentState().setColorWriteMask(
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)};
	const auto pipelineColorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo()
		.setAttachmentCount(attachState.size())
		.setPAttachments(attachState.data())
		.setLogicOp(vk::LogicOp::eNoOp)
		.setBlendConstants({1.0f, 1.0f, 1.0f, 1.0f});

	// Pipeline viewport state
	const auto vpCreateStateInfo = vk::PipelineViewportStateCreateInfo().setViewportCount(1).setScissorCount(1);

	// Add viewport and scissor to dynamic states (to be able to resize window
	// without re-creating pipeline
	std::vector<vk::DynamicState> dynamicStates;
	dynamicStates.push_back(vk::DynamicState::eViewport);
	dynamicStates.push_back(vk::DynamicState::eScissor);
	// Dynamic state, changeable by command buffer commands
	auto dynamicStateCreateInfo = vk::PipelineDynamicStateCreateInfo().setDynamicStateCount(dynamicStates.size()).setPDynamicStates(dynamicStates.data());

	// Pipeline Depth Stencil State
	const auto depthStencilState = vk::PipelineDepthStencilStateCreateInfo()
		.setDepthTestEnable(VK_TRUE)
		.setDepthWriteEnable(VK_TRUE)
		.setDepthCompareOp(vk::CompareOp::eLessOrEqual)
		.setBack(vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways))
		.setFront(vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways));

	// No multisample for now
	const auto pipelineMultisampleStateCreateInfo = vk::PipelineMultisampleStateCreateInfo();

	vk::PipelineCacheCreateInfo pipelineCache;
	info.pipelineCache = info.device.createPipelineCache(pipelineCache);

	// Create the pipeline from all the pipeline states
	const auto graphicsPipelineCreateInfo = vk::GraphicsPipelineCreateInfo()
		.setLayout(info.pipelineLayout)
		.setPVertexInputState(&vertexInputStateCreateInfo)
		.setPInputAssemblyState(&vertexInputAssemblyStateCreateInfo)
		.setPRasterizationState(&pipelineRasterStateCreateInfo)
		.setPColorBlendState(&pipelineColorBlendStateCreateInfo)
		.setPMultisampleState(&pipelineMultisampleStateCreateInfo)
		.setPDynamicState(&dynamicStateCreateInfo)
		.setPViewportState(&vpCreateStateInfo)
		.setPDepthStencilState(&depthStencilState)
		.setPStages(info.shaderStages.data())
		.setStageCount(info.shaderStages.size())
		.setRenderPass(info.renderPass);

	auto pipeline = info.device.createGraphicsPipeline(info.pipelineCache, graphicsPipelineCreateInfo);

    switch ( pipeline.result )
    {
      case vk::Result::eSuccess: break;
      case vk::Result::ePipelineCompileRequiredEXT:
		std::cout << "Pipeline requires extension" << std::endl;
        break;
      default: assert( false );  // should never happen
    }

	info.pipeline = pipeline.value;

	return 0;
}

std::pair<vk::Semaphore, vk::Fence> recordCommandBuffer() {
	// Create semaphore for waiting for the previous frame
	const auto imageAquiredSemaphoreCreateInfo = vk::SemaphoreCreateInfo();
	const auto imageAquiredSemaphore = info.device.createSemaphore(imageAquiredSemaphoreCreateInfo);

	// Get the index of the next available swapchain image
	info.currentBuffer = info.device.acquireNextImageKHR(info.swapchain, UINT64_MAX, imageAquiredSemaphore, nullptr).value;

	// Use the current CommandBuffer
	auto &cmd = info.cmdBuffers.front();

	// Begin the render pass
	const auto clearValues =
		std::vector<vk::ClearValue>{vk::ClearValue().setColor(std::array<float, 4>{0.2f, 0.2f, 0.2f, 0.2f}), vk::ClearValue().setDepthStencil({1.0f, 0})};
	const auto renderPassBegin = vk::RenderPassBeginInfo()
		.setRenderPass(info.renderPass)
		.setFramebuffer(info.framebuffers[info.currentBuffer])
		.setRenderArea({{0, 0}, info.surfaceCapabilities.currentExtent})
		.setClearValueCount(2)
		.setPClearValues(clearValues.data());
	cmd.beginRenderPass(renderPassBegin, vk::SubpassContents::eInline);

	// Bind the pipeline to the command buffer
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, info.pipeline);

	// Bind the descriptor sets
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, info.pipelineLayout, 0, info.descriptorSets.size(), info.descriptorSets.data(), 0, nullptr);
	// cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
	// info.pipelineLayout, 0, info.descriptorSets);

	// Bind the vertex buffer
	cmd.bindVertexBuffers(0, info.vertexBufferData.buffer, {0});

	// Set viewport and scissor
	const auto viewport = vk::Viewport()
		.setHeight(info.surfaceCapabilities.currentExtent.height)
		.setWidth(info.surfaceCapabilities.currentExtent.width)
		.setMinDepth(0.0f)
		.setMaxDepth(1.0f)
		.setX(0)
		.setY(0);
	cmd.setViewport(0, 1, &viewport);
	const auto scissor = vk::Rect2D().setExtent(info.surfaceCapabilities.currentExtent).setOffset({0, 0});
	cmd.setScissor(0, 1, &scissor);

	// The actual draw command
	cmd.draw(12 * 3, 1, 0, 0);

	// End the render pass
	cmd.endRenderPass();

	endCommandBuffer();

	// First create a fence to know when the GPU is done
	const auto fenceCreateInfo = vk::FenceCreateInfo();
	const auto drawFence = info.device.createFence(fenceCreateInfo);

	// Then submit command buffer
	vk::PipelineStageFlags pipelineStageFlags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	const auto submitInfo = vk::SubmitInfo()
		.setWaitSemaphoreCount(1)
		.setPWaitSemaphores(&imageAquiredSemaphore)
		.setCommandBufferCount(1)
		.setPCommandBuffers(info.cmdBuffers.data())
		.setPWaitDstStageMask(&pipelineStageFlags);
	info.gfxQueue.submit(submitInfo, drawFence);

	vk::Result res;
	do {
		res = info.device.waitForFences(drawFence, VK_TRUE, 1000000);
	} while (res == vk::Result::eTimeout);

	return {imageAquiredSemaphore, drawFence};
}

void presentSwapChainImage() {
	const auto presentInfo = vk::PresentInfoKHR().setSwapchainCount(1).setPSwapchains(&info.swapchain).setPImageIndices(&info.currentBuffer);
	info.prsntQueue.presentKHR(presentInfo);
}

void destroy_pipeline(vkinfo &info) { info.device.destroyPipeline(info.pipeline); }

void destroy_pipeline_cache(vkinfo &info) { info.device.destroyPipelineCache(info.pipelineCache); }

void destroy_uniform_buffer(vkinfo &info) {
	info.device.destroyBuffer(info.uniformData.buffer);
	info.device.freeMemory(info.uniformData.mem);
}

void destroy_descriptor_and_pipeline_layouts(vkinfo &info) {
	for (unsigned int i = 0; i < info.descriptorSets.size(); i++)
		info.device.destroyDescriptorSetLayout(info.descriptorSetLayouts[i]);
	info.device.destroyPipelineLayout(info.pipelineLayout);
}

void destroy_descriptor_pool(vkinfo &info) { info.device.destroyDescriptorPool(info.descriptorPool); }

void destroy_shaders(vkinfo &info) {
	info.device.destroyShaderModule(info.shaderStages[0].module);
	info.device.destroyShaderModule(info.shaderStages[1].module);
}

void destroy_command_buffer(vkinfo &info) { info.device.freeCommandBuffers(info.cmdPool, info.cmdBuffers); }

void destroy_command_pool(vkinfo &info) { info.device.destroyCommandPool(info.cmdPool); }

void destroy_depth_buffer(vkinfo &info) {
	info.device.destroyImageView(info.depth.view);
	info.device.destroyImage(info.depth.image);
	info.device.freeMemory(info.depth.mem);
}

void destroy_vertex_buffer(vkinfo &info) {
	info.device.destroyBuffer(info.vertexBufferData.buffer);
	info.device.freeMemory(info.vertexBufferData.mem);
}

void destroy_swap_chain(vkinfo &info) {
	for (uint32_t i = 0; i < info.swapchainBuffers.size(); i++) {
		info.device.destroyImageView(info.swapchainBuffers[i].view);
	}
	info.device.destroy(info.swapchain);
}

void destroy_framebuffers(vkinfo &info) {
	for (uint32_t i = 0; i < info.framebuffers.size(); i++) {
		info.device.destroyFramebuffer(info.framebuffers[i]);
	}
	info.framebuffers.clear();
}

void destroy_renderpass(vkinfo &info) { info.device.destroyRenderPass(info.renderPass); }

void destroy_device(vkinfo &info) {
	info.device.waitIdle();
	info.device.destroy();
}

void destroy_instance(vkinfo &info) { info.inst.destroy(); }

void destroy_window(vkinfo &info) {
	info.inst.destroy(info.surface);
	SDL_DestroyWindow(sdlinfo.window);
}

// void destroy_textures(vkinfo &info)
//{
//  for(size_t i = 0; i < info.textures.size(); i++) {
//    vkDestroySampler(info.device, info.textures[i].sampler, NULL);
//    vkDestroyImageView(info.device, info.textures[i].view, NULL);
//    vkDestroyImage(info.device, info.textures[i].image, NULL);
//    vkFreeMemory(info.device, info.textures[i].image_memory, NULL);
//    vkDestroyBuffer(info.device, info.textures[i].buffer, NULL);
//    vkFreeMemory(info.device, info.textures[i].buffer_memory, NULL);
//  }
//}

int main() {
#ifdef WIN32
	std::cout.rdbuf(&g_DebugStreamFor_cout); // Redirect std::cout to OutputDebugString!
#endif

	setupApplicationAndInstance();
	setupDevicesAndQueues();
	setupCommandBuffers();
	setupSwapChains();
	setupDepth();

	const auto proj = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
	const auto view = glm::lookAt(glm::vec3(-5, 3, -10), // Camera is at (-5,3,-10), in World Space
			glm::vec3(0, 0, 0),    // and looks at the origin
			glm::vec3(0, -1, 0));  // Head is up (set to 0,-1,0 to look upside-down)
	const auto model = glm::mat4(1.0f);
	// Vulkan clip space has inverted Y and half Z.
	const auto clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f);
	const auto mvp = clip * proj * view * model;

	setupUniformBuffer((void *)&mvp, sizeof(mvp));

	setupDescriptorSetLayoutAndPipelineLayout();
	setupDescriptorSetPool();
	allocateDescriptorSet();
	initRenderPass();
	setupShaders();
	setupFrameBuffers();
	createVertexBuffer();
	setupPipelineStates();

	vk::Semaphore imageAcquiredSemaphore;
	vk::Fence drawFence;

	beginCommandBuffer();
	auto res = recordCommandBuffer();
	imageAcquiredSemaphore = res.first;
	drawFence = res.second;
	// endCommandBuffer();
	// submitCommandBuffer();

	// Poll for user input.
	bool stillRunning = true;
	presentSwapChainImage();
	while (stillRunning) {

		SDL_Event event;
		while (SDL_PollEvent(&event)) {

			switch (event.type) {

				case SDL_QUIT:
					stillRunning = false;
					break;

				default:

					break;
			}
		}

		SDL_Delay(10);
	}

	// Clean up.
	info.device.destroySemaphore(imageAcquiredSemaphore);
	info.device.destroyFence(drawFence);
	destroy_pipeline(info);
	destroy_pipeline_cache(info);
	destroy_descriptor_pool(info);
	destroy_vertex_buffer(info);
	destroy_framebuffers(info);
	destroy_shaders(info);
	destroy_renderpass(info);
	destroy_descriptor_and_pipeline_layouts(info);
	destroy_uniform_buffer(info);
	destroy_depth_buffer(info);
	destroy_swap_chain(info);
	destroy_command_buffer(info);
	destroy_command_pool(info);
	destroy_device(info);
	destroy_window(info);
	destroy_instance(info);

	SDL_Quit();

	return 0;
}

vk::SurfaceKHR createVulkanSurface(const vk::Instance &instance, SDL_Window *window) {
	SDL_SysWMinfo windowInfo;
	SDL_VERSION(&windowInfo.version);
	if (!SDL_GetWindowWMInfo(window, &windowInfo)) {
		throw std::system_error(std::error_code(), "SDK window manager info is not available.");
	}

	switch (windowInfo.subsystem) {

#if defined(SDL_VIDEO_DRIVER_ANDROID) && defined(VK_USE_PLATFORM_ANDROID_KHR)
		case SDL_SYSWM_ANDROID: {
									vk::AndroidSurfaceCreateInfoKHR surfaceInfo = vk::AndroidSurfaceCreateInfoKHR().setWindow(windowInfo.info.android.window);
									return instance.createAndroidSurfaceKHR(surfaceInfo);
								}
#endif

#if defined(SDL_VIDEO_DRIVER_WAYLAND) && defined(VK_USE_PLATFORM_WAYLAND_KHR)
		case SDL_SYSWM_WAYLAND: {
									vk::WaylandSurfaceCreateInfoKHR surfaceInfo =
										vk::WaylandSurfaceCreateInfoKHR().setDisplay(windowInfo.info.wl.display).setSurface(windowInfo.info.wl.surface);
									return instance.createWaylandSurfaceKHR(surfaceInfo);
								}
#endif

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && defined(VK_USE_PLATFORM_WIN32_KHR)
		case SDL_SYSWM_WINDOWS: {
									vk::Win32SurfaceCreateInfoKHR surfaceInfo = vk::Win32SurfaceCreateInfoKHR().setHinstance(GetModuleHandle(NULL)).setHwnd(windowInfo.info.win.window);
									return instance.createWin32SurfaceKHR(surfaceInfo);
								}
#endif

#if defined(SDL_VIDEO_DRIVER_X11) && defined(VK_USE_PLATFORM_XLIB_KHR)
		case SDL_SYSWM_X11: {
								vk::XlibSurfaceCreateInfoKHR surfaceInfo = vk::XlibSurfaceCreateInfoKHR().setDpy(windowInfo.info.x11.display).setWindow(windowInfo.info.x11.window);
								return instance.createXlibSurfaceKHR(surfaceInfo);
							}
#endif

		default:
							throw std::system_error(std::error_code(), "Unsupported window manager is in use.");
	}
}

std::vector<const char *> getAvailableWSIExtensions() {
	std::vector<const char *> extensions;
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
