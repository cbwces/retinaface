# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cbw233/src/retinaface

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cbw233/src/retinaface/build

# Include any dependencies generated for this target.
include CMakeFiles/retinaface.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/retinaface.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/retinaface.dir/flags.make

CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.o: CMakeFiles/retinaface.dir/flags.make
CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.o: ../retinaface/RetinaFace.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cbw233/src/retinaface/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.o -c /home/cbw233/src/retinaface/retinaface/RetinaFace.cpp

CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cbw233/src/retinaface/retinaface/RetinaFace.cpp > CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.i

CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cbw233/src/retinaface/retinaface/RetinaFace.cpp -o CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.s

CMakeFiles/retinaface.dir/retinaface/main.cpp.o: CMakeFiles/retinaface.dir/flags.make
CMakeFiles/retinaface.dir/retinaface/main.cpp.o: ../retinaface/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cbw233/src/retinaface/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/retinaface.dir/retinaface/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/retinaface.dir/retinaface/main.cpp.o -c /home/cbw233/src/retinaface/retinaface/main.cpp

CMakeFiles/retinaface.dir/retinaface/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/retinaface.dir/retinaface/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cbw233/src/retinaface/retinaface/main.cpp > CMakeFiles/retinaface.dir/retinaface/main.cpp.i

CMakeFiles/retinaface.dir/retinaface/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/retinaface.dir/retinaface/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cbw233/src/retinaface/retinaface/main.cpp -o CMakeFiles/retinaface.dir/retinaface/main.cpp.s

# Object files for target retinaface
retinaface_OBJECTS = \
"CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.o" \
"CMakeFiles/retinaface.dir/retinaface/main.cpp.o"

# External object files for target retinaface
retinaface_EXTERNAL_OBJECTS =

retinaface: CMakeFiles/retinaface.dir/retinaface/RetinaFace.cpp.o
retinaface: CMakeFiles/retinaface.dir/retinaface/main.cpp.o
retinaface: CMakeFiles/retinaface.dir/build.make
retinaface: CMakeFiles/retinaface.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cbw233/src/retinaface/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable retinaface"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/retinaface.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/retinaface.dir/build: retinaface

.PHONY : CMakeFiles/retinaface.dir/build

CMakeFiles/retinaface.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/retinaface.dir/cmake_clean.cmake
.PHONY : CMakeFiles/retinaface.dir/clean

CMakeFiles/retinaface.dir/depend:
	cd /home/cbw233/src/retinaface/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cbw233/src/retinaface /home/cbw233/src/retinaface /home/cbw233/src/retinaface/build /home/cbw233/src/retinaface/build /home/cbw233/src/retinaface/build/CMakeFiles/retinaface.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/retinaface.dir/depend
