#!/usr/bin/env python3

""" ValidateShaders.py: Compiles and validates GLSL shaders
    using Khronos official reference compiler."""

__author__ = "Peter Asplund"
__credits__ = ["Khronos"]
__version__ = "0.3"
__maintainer__ = "Peter Asplund"
__email__ = "peter.asplund@raysearchlabs.com"
__status__ = "Production"

import os
import re
from optparse import OptionParser
from subprocess import check_output
from subprocess import CalledProcessError
from collections import namedtuple
from colorama import init, Fore, Style
init()  # init colorama

Shader = namedtuple('Shader',
                    ['name',
                     'vert_shader',
                     'geom_shader',
                     'frag_shader',
                     'has_main'])


def read_include_file(line):
    """ Read the entire file and return the string """
    file_name = line[10:-1].strip()
    included_shader = ""
    with open(file_name, 'r') as include_file:
        for include_line in include_file.read().splitlines():
            if include_line.find('#include') != -1:
                included_shader += read_include_file(include_line)
            else:
                included_shader += include_line + '\n'
    return included_shader


def read_shaders(shader_file):
    """ Read the file and return a shader object """
    with open(shader_file, 'r') as file_handle:

        # Break up glsl file into each respective shader
        new_shader = dict(common_string="",
                          vert_shader="",
                          geom_shader="",
                          frag_shader="")

        mode = "common_string"
        for line in file_handle.read().splitlines():
            line = line.rstrip()
            if line == '@common':
                mode = "common_string"
                continue
            elif line == '@vert':
                mode = "vert_shader"
                continue
            elif line == '@geom':
                mode = "geom_shader"
                continue
            elif line == '@frag':
                mode = "frag_shader"
                continue

            include_found = line.find('#include') != -1

            if include_found:
                new_shader[mode] += read_include_file(line)
                continue
            new_shader[mode] += line + '\n'

        has_main = False
        if (new_shader["common_string"].find('void main') != -1 or
                new_shader["vert_shader"].find('void main') != -1 or
                new_shader["geom_shader"].find('void main') != -1 or
                new_shader["frag_shader"].find('void main') != -1):
            has_main = True

        name = shader_file[shader_file.rfind('\\') + 1:]
        shader = Shader(name,
                        (new_shader["common_string"] +
                         new_shader["vert_shader"]).strip(),
                        (new_shader["common_string"] +
                         new_shader["geom_shader"]).strip(),
                        (new_shader["common_string"] +
                         new_shader["frag_shader"]).strip(),
                        has_main)
    return shader


def parse_cmd_arguments():
    """ Parse command line arguments """
    # Parse command line commands
    parser = OptionParser()
    parser.add_option("-v", "--verbose", action="store_true",
                      dest="verbose",
                      help="Verbose logging", default=False)
    parser.add_option("-b", "--break", action="store_true",
                      dest="break_on_error",
                      help="Break on error, leaving temporary files.",
                      default=False)
    parser.add_option("-f", "--file", dest="file_name",
                      help="File to parse and compile",
                      metavar="FILE")

    (options, args) = parser.parse_args()

    if args:
        print("Unkown argument: " + args)
        exit(-1)

    return options


def find_failing_lines(exception_message):
    """ Parse which line is failing and return the line numbers """
    failing_lines = []
    key = re.compile(br'\d+\:\d+', re.IGNORECASE)
    for line in exception_message.splitlines():
        numbers = key.findall(line)
        if numbers:
            idxs = re.findall(br'\d+', numbers[0])
            failing_lines.append(int(idxs[1]))
    return failing_lines


def print_shader(shader_code, failing_lines):
    """ Print the shader source code and highlight failing lines """
    line_nr = 1
    print()
    for line in shader_code.splitlines():
        if line_nr in failing_lines:
            print(Style.BRIGHT + Fore.RED + ' ' + str(line_nr) + '\t' +
                  line + Style.RESET_ALL + Fore.RESET)
        else:
            print(' ' + str(line_nr) + '\t' + line)
        line_nr += 1


def validate_shader(validation_command, shader_name, shader_code,
                    shader_file, break_on_error):
    """ Run validation_command and output formatted error if failed """
    failed_validation = 0
    break_validation = False
    try:
        check_output(validation_command)
    except (CalledProcessError) as exception:
        print('\nValidation of ' + shader_name +
              ' failed. ' + shader_file + ' :\n' + Style.BRIGHT +
              exception.output.strip().decode('ascii') + Style.RESET_ALL)
        print_shader(shader_code,
                     find_failing_lines(exception.output.strip()))
        failed_validation = 1
        if break_on_error:
            print("Break on error enabled. Bailing. \
                   Temporary file is called" + shader_file)
            break_validation = True
    return (failed_validation, break_validation)


def validate_shaders(options, shader, failed_validation):
    """ Validates the shader and prints error code if failed """
    break_validation = False

    # Command to call later
    validation_command = ['glslangValidator.exe']
    arguments = []
    # Check which shaders exist
    if shader.vert_shader.find('void main') != -1:
        arguments.append((shader.vert_shader, 'tmpshader.vert'))
    if shader.geom_shader.find('void main') != -1:
        arguments.append((shader.geom_shader, 'tmpshader.geom'))
    if shader.frag_shader.find('void main') != -1:
        arguments.append((shader.frag_shader, 'tmpshader.frag'))

    # Validate compilation of each separate shader
    for (shader_code, shader_file) in arguments:
        (failed, break_validation) = validate_shader(
            validation_command + [shader_file],
            shader.name,
            shader_code,
            shader_file,
            options.break_on_error)
        failed_validation += failed

    # Validate linking of all files together
    validation_command.append('-V')
    # Build list only of tmpshader names
    shader_files = [stage[1] for stage in arguments]
    validation_command.extend(shader_files)
    (failed, break_validation) = validate_shader(validation_command,
                                                 shader.name,
                                                 '',
                                                 'Linking stage',
                                                 options.break_on_error)

    try:
        # Clean-up temporary files
        os.remove('tmpshader.vert')
        os.remove('tmpshader.geom')
        os.remove('tmpshader.frag')
    except OSError as exception:
        print('Unable to remove temporary files', exception.strerror)

    return (failed_validation, break_validation)


def do_main_program():
    """ Main program """
    options = parse_cmd_arguments()

    print("\nValidating GLSL Shaders using Khronos glslangValidator")
    found_files = [""]
    if options.file_name:
        found_files = [options.file_name]
    else:
        found_files = [
            os.path.join(os.getcwd(), f)
            for f in os.listdir('.')
            if os.path.isfile(os.path.join(os.getcwd(), f))
            if f.endswith(".glsl")
        ]

    validated = 0
    failed_validation = 0
    for found_file in found_files:
        shader = read_shaders(found_file)

        # If the file has no main, we won't evaluate it
        if not shader.has_main:
            continue

        validated += 1

        # Write to temporary file and validate it
        with open("tmpshader.vert", 'w') as file_handle:
            file_handle.write(shader.vert_shader)
        with open("tmpshader.geom", 'w') as file_handle:
            file_handle.write(shader.geom_shader)
        with open("tmpshader.frag", 'w') as file_handle:
            file_handle.write(shader.frag_shader)

        # Validate shaders and print output from validator if failed
        (failed_validation, break_validation) = validate_shaders(
            options,
            shader,
            failed_validation)

        if break_validation:
            break

    print("\n" + str(validated) + " files in directory validated")
    print(str(failed_validation) + " files failed validation")
    if failed_validation:
        exit(-1)


# Main function
try:
    if __name__ == "__main__":
        do_main_program()
except KeyboardInterrupt:
    print("Keyboard Interrupted!")
