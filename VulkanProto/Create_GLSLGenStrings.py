# Imports
from os import listdir
import os, stat

# Constants
startOfFileStr = """
// Automatic generated file that contains code from glsl program files.
// All manual changes will be deleted when file is regenerated.
// This file should NOT be under version control.

#include "Pch.h"
#include "GLSLStrings.h"
#include <sstream>
#include <string>

using namespace std;

RaySearch::CoreVTK::gl::GLSLStrings::GLSLStrings()
{

"""

endOfFileStr = "}"

# Returns a string with quotes around
def stringify(str):
  return "\"" + str + "\""
  
# Write content to this string, using stringstream as we might reach the max length for a string
cxxFileStr = startOfFileStr
for path in listdir("."):
  if not path.endswith(".glsl"):
    continue
  file = open(path, 'r')
  fileName = file.name
  str = ""
  str += "  // " + fileName + "\n"
  str += "  if (programStrings_.count(" + stringify(fileName) + ") == 0)\n"
  str += "  {\n"
  str += "    stringstream sstr;\n"
  for line in file:
    line = line.replace("\t", "  ").rstrip().ljust(60)
    str += "    sstr << \"  " + line + "\" << endl;\n"
  str += "    programStrings_[" + stringify(fileName) + "] = sstr.str();\n"
  str += "  }\n\n"
  cxxFileStr += str
cxxFileStr += endOfFileStr

# Remove write attribute and write to file
outFilename = "GLSLGenStrings.cxx"
if os.path.isfile(outFilename):
  os.chmod(outFilename, stat.S_IWRITE)
outFile = open(outFilename, 'w')
outFile.write(cxxFileStr)
outFile.close()
