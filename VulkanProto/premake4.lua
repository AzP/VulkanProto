solution "VulkanProto"
	location("./")
	targetdir("./")
	configurations { "Debug", "Release" }
	platforms { "x64" }
	objdir("obj/" .. os.get() .. "/")
	local sourcedir = "./"

	project "VulkanProto"
		--kind "WindowedApp"
		kind "ConsoleApp"
		language "C++"
		files { sourcedir .. "*.cpp", sourcedir .. "*.h" }

		configuration "windows"
			includedirs {"./include", "./include/freetype2"}
			configuration "x64"
				libdirs {"./lib/x64"}
			configuration "x86"
				libdirs {"./lib/x86"}

		configuration "linux"
			libdirs {"/usr/lib", "/usr/local/lib"}
			includedirs {"/usr/include", "/usr/include/vulkan/", "/usr/include/freetype2", "/usr/include/tmxparser/"}
		
		configuration "Debug"
			targetname ("VulkanProto-debug")
			defines { "_USE_MATH_DEFINES", "WAFFLE_API_VERSION=0x0103", "DEBUG"}
			flags { "Symbols", "ExtraWarnings", "FatalWarnings" }
			configuration "windows"
				links {"vulkan","DevIL","ILU","ILUT","SDL2","SDL2main","freetype264d","tmxparser"}
				-- disablewarnings { "4668;4201;4290;4522" } -- Not yet supported in premake4
			configuration "linux"
				links {"vulkan","IL","ILU","SDL2","ubsan","freetype","tmxparser"}
				buildoptions{ "-std=c++17 -Wpedantic -Wall -g3 -O0 -Wextra -fsanitize=undefined -fno-sanitize-recover -fsanitize=shift -fsanitize=integer-divide-by-zero -fsanitize=unreachable -fsanitize=vla-bound -fsanitize=null -fsanitize=return -fsanitize=signed-integer-overflow -fsanitize=bounds -fsanitize=alignment -fsanitize=object-size -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fsanitize=nonnull-attribute -fsanitize=returns-nonnull-attribute -fsanitize=bool -fsanitize=enum -fsanitize=vptr" }

		configuration "Release"
			targetname ("VulkanProto-release")
			defines { "_USE_MATH_DEFINES", "WAFFLE_API_VERSION=0x0103", "NDEBUG"}
			flags { "Optimize", "ExtraWarnings", "FatalWarnings" }
			configuration "windows"
				links {"vulkan","DevIL","ILU","ILUT","SDL2","SDL2main","freetype264d","tmxparser"}
				-- undefines{ "_UNICODE" }
				-- disablewarnings { "4668;4201;4290;4522" }
			configuration "linux"
				links {"vulkan","IL","ILU","SDL2","ubsan","freetype","tmxparser"}
				buildoptions{ "-std=c++17 -Wpedantic -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover -fsanitize=shift -fsanitize=integer-divide-by-zero -fsanitize=unreachable -fsanitize=vla-bound -fsanitize=null -fsanitize=return -fsanitize=signed-integer-overflow -fsanitize=bounds -fsanitize=alignment -fsanitize=object-size -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fsanitize=nonnull-attribute -fsanitize=returns-nonnull-attribute -fsanitize=bool -fsanitize=enum -fsanitize=vptr" }

