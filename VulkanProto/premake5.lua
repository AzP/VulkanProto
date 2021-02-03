solution "VulkanProto"
	location("./")
	targetdir("./bin/")
	configurations { "Debug", "Release" }
	platforms { "x64" }
	objdir("obj/" .. os.target() .. "/")
	local sourcedir = "./"

	project "VulkanProto"
		--kind "WindowedApp"
		kind "ConsoleApp"
		language "C++"
		cppdialect "C++17"
		files { sourcedir .. "*.cpp", sourcedir .. "*.h" }

		configuration "windows"
			includedirs {"./include", "./include/freetype2","$(VULKAN_SDK)/include" }
			configuration "x64"
				libdirs {"./lib/x64"}

		configuration "linux"
			libdirs {"/usr/lib", "/usr/local/lib"}
			includedirs {"/usr/include", "/usr/include/SDL2", "/usr/include/vulkan/"}
		
		configuration "Debug"
			targetname ("VulkanProto-debug")
			defines { "_USE_MATH_DEFINES", "DEBUG"}
			symbols "On"
			warnings "Extra"
			configuration "windows"
				links { "$(VULKAN_SDK)/lib/vulkan-1.lib","SDL2"}
				disablewarnings { "4668;4201;4290;4522" } 
			configuration "linux"
				links {"vulkan","SDL2","ubsan"}
				buildoptions{ "-std=c++17 -Wpedantic -Wall -g3 -O0 -Wextra -fsanitize=undefined -fno-sanitize-recover -fsanitize=shift -fsanitize=integer-divide-by-zero -fsanitize=unreachable -fsanitize=vla-bound -fsanitize=null -fsanitize=return -fsanitize=signed-integer-overflow -fsanitize=bounds -fsanitize=alignment -fsanitize=object-size -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fsanitize=nonnull-attribute -fsanitize=returns-nonnull-attribute -fsanitize=bool -fsanitize=enum -fsanitize=vptr" }

		configuration "Release"
			targetname ("VulkanProto-release")
			defines { "_USE_MATH_DEFINES", "NDEBUG"}
			warnings "Extra"
			optimize "On"
			configuration "windows"
				links {"$(VULKAN_SDK)/lib/vulkan-1.lib","SDL2"}
				-- undefines{ "_UNICODE" }
				disablewarnings { "4668;4201;4290;4522" }
			configuration "linux"
				links {"vulkan","SDL2","ubsan"}
				buildoptions{ "-std=c++17 -Wpedantic -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover -fsanitize=shift -fsanitize=integer-divide-by-zero -fsanitize=unreachable -fsanitize=vla-bound -fsanitize=null -fsanitize=return -fsanitize=signed-integer-overflow -fsanitize=bounds -fsanitize=alignment -fsanitize=object-size -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fsanitize=nonnull-attribute -fsanitize=returns-nonnull-attribute -fsanitize=bool -fsanitize=enum -fsanitize=vptr" }

