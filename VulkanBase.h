#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <set>
#include <array>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>


class VulkanBase {
public:
    VulkanBase(int w, int h);
	void run(void *handle, VkExternalMemoryHandleTypeFlagBits handleType, size_t size);
    VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType();
    void createExternalSemaphore(VkSemaphore& semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType);

private:

	const uint32_t WIDTH;
	const uint32_t HEIGHT;

	const int MAX_FRAMES_IN_FLIGHT;

	const std::vector<const char*> validationLayers = {
	    "VK_LAYER_KHRONOS_validation"
	};

	std::vector<const char*> deviceExtensions = {
	    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        // VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        // VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        // VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        // VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
	};

	const bool enableValidationLayers;

    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    VkSemaphore m_vkWaitSemaphore, m_vkSignalSemaphore;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    size_t currentFrame;

    bool framebufferResized;

    VkBuffer cpyBuffer;
    VkDeviceMemory cpyBufferMemory;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkFormat format;

	struct QueueFamilyIndices {
	    std::optional<uint32_t> graphicsFamily;
	    std::optional<uint32_t> presentFamily;

	    bool isComplete() {
	        return graphicsFamily.has_value() && presentFamily.has_value();
	    }
	};

	struct SwapChainSupportDetails {
	    VkSurfaceCapabilitiesKHR capabilities;
	    std::vector<VkSurfaceFormatKHR> formats;
	    std::vector<VkPresentModeKHR> presentModes;
	};

	struct Vertex {
	    glm::vec2 pos;
	    glm::vec3 color;
	    glm::vec2 texCoord;

	    static VkVertexInputBindingDescription getBindingDescription() {
	        VkVertexInputBindingDescription bindingDescription{};
	        bindingDescription.binding = 0;
	        bindingDescription.stride = sizeof(Vertex);
	        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	        return bindingDescription;
	    }

	    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
	        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

	        attributeDescriptions[0].binding = 0;
	        attributeDescriptions[0].location = 0;
	        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
	        attributeDescriptions[0].offset = offsetof(Vertex, pos);

	        attributeDescriptions[1].binding = 0;
	        attributeDescriptions[1].location = 1;
	        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	        attributeDescriptions[1].offset = offsetof(Vertex, color);

	        attributeDescriptions[2].binding = 0;
	        attributeDescriptions[2].location = 2;
	        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
	        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

	        return attributeDescriptions;
	    }
	};

	const std::vector<Vertex> vertices = {
	    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
	    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
	    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
	};

	const std::vector<uint16_t> indices = {
	    0, 1, 2, 2, 3, 0
	};

	struct UniformBufferObject {
	    alignas(16) glm::mat4 model;
	    alignas(16) glm::mat4 view;
	    alignas(16) glm::mat4 proj;
	};

    void initWindow();
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    void initVulkan(void *handle, VkExternalMemoryHandleTypeFlagBits handleType, size_t size);
    void mainLoop();
    void cleanupSwapChain();
    void cleanup();
    void recreateSwapChain();
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createTextureImage(void *handle, VkExternalMemoryHandleTypeFlagBits handleType, size_t size,
        VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void createTextureImageView();
    VkImageView createImageView(VkImage image, VkFormat format);
    void createTextureSampler();
    void createVertexBuffer();
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
    	VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();
    void drawFrame();
    void updateUniformBuffer(uint32_t currentImage);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    std::vector<const char*> getRequiredExtensions();
    bool checkValidationLayerSupport();
    static std::vector<char> readFile(const std::string& filename);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    	VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

};