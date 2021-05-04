#!/usr/bin/env python3

""" validate_shaders.py: Compiles and validates GLSL shaders
    using Khronos official reference compiler."""

__author__ = "Peter Asplund"
__credits__ = ["Khronos"]
__version__ = "0.4"
__maintainer__ = "Peter Asplund"
__email__ = "peter.azp@gmail.com"
__status__ = "Production"

import os
import re
import sys
import argparse
import logging
from subprocess import run
from subprocess import CalledProcessError
from collections import namedtuple

# colorama dependency
from colorama import init, Fore, Style
init()  # init colorama

Shader = namedtuple('Shader',
                    ['name',
                     'vert_shader',
                     'geom_shader',
                     'frag_shader',
                     'has_main']
                    )


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
            if line == '@vert':
                mode = "vert_shader"
                continue
            if line == '@geom':
                mode = "geom_shader"
                continue
            if line == '@frag':
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
                        dest="verbose",
                        help="Verbose logging", default=False)
    parser.add_argument("-b", "--break", action="store_true",
                        dest="break_on_error",
                        help="Break on error, leaving temporary files.",
                        default=False)
    parser.add_argument("-f", "--file", dest="file_name",
                        help="File to parse and compile",
                        metavar="FILE")
    parser.add_argument("-u", "--unify", action="store_true",
                        dest="unify_into_one_file",
                        help="Unify spv stages into one spv file",
                        default=False)
    parser.add_argument("-d", "--debug", action="store_true",
                        dest="debug",
                        help="Print debug information to console",
                        default=False)
    options = parser.parse_args()
    return options


def setup_logging(options):
    """ Setup logging to tty. """
    # Logging format for logfile and console messages
    # formatting = "%(asctime)s (%(process)d) %(levelname)s: %(message)s"
    formatting = "%(message)s"

    if options.debug:
        logging.basicConfig(format=formatting, level=logging.DEBUG)
    elif options.verbose:
        logging.basicConfig(format=formatting, level=logging.INFO)
    else:
        logging.basicConfig(format=formatting, level=logging.CRITICAL)

    logging.Formatter(formatting)


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


def run_command(validation_command, shader_name, shader_code,
                shader_file, break_on_error):
    """ Run validation_command and output formatted error if failed """
    failed_validation = 0
    break_validation = False
    try:
        logging.debug(validation_command)
        result = run(validation_command,
                     shell=False, check=True, text=True, capture_output=True)
        if len(result.stdout) > 0:
            logging.debug(result.stdout)
    except (CalledProcessError) as exception:
        logging.info('\nValidation of' + shader_name +
                     'failed.' + shader_file + ':\n' + Style.BRIGHT +
                     exception.output.strip().decode() + Style.RESET_ALL)
        print_shader(shader_code,
                     find_failing_lines(exception.output.strip()))
        failed_validation = 1
        if break_on_error:
            logging.info("Break on error enabled. Bailing. \
                         Temporary file is called %s", shader_file)
            break_validation = True
    return (failed_validation, break_validation)


def identify_shader_stages(shader):
    """ Identify which shader stages to validate """
    shader_stages = []
    # Check which shaders exist
    if shader.vert_shader.find('void main') != -1:
        shader_stages.append((shader.vert_shader, 'tmpshader.vert'))
    if shader.geom_shader.find('void main') != -1:
        shader_stages.append((shader.geom_shader, 'tmpshader.geom'))
    if shader.frag_shader.find('void main') != -1:
        shader_stages.append((shader.frag_shader, 'tmpshader.frag'))
    return shader_stages


def validate_compilation(validation_command, shader, shader_stages, options):
    """ Validate compilation of each shader stage.
        Return number of failed stages. """
    failed_validation = 0
    for (shader_code, shader_filename) in shader_stages:
        (failed, break_validation) = run_command(
                validation_command + [shader_filename],
                shader.name,
                shader_code,
                shader_filename,
                options.break_on_error)
        failed_validation += failed
        if break_validation:
            break
    return (failed, break_validation)


def validate_linking(validation_command, shader, shader_stages, options):
    """ Validate linking of all shader stages. Return 1 if failed """
    # Build list only of tmpshader names and validate linking
    shader_files = [stage[1] for stage in shader_stages]
    link_validation_command = validation_command[:]
    link_validation_command.append('-l')
    link_validation_command.extend(shader_files)
    (failed, break_validation) = run_command(link_validation_command,
                                             shader.name,
                                             '',
                                             'Validate linking stage',
                                             options.break_on_error)
    return (failed, break_validation)


def generate_spirv_stages(validation_command, shader_name,
                          shader, shader_stages, options):
    """ Create single SPIR-V module per file """
    failed_validation = 0
    spv_stages = []
    spv_output_dir_name = './spv/'
    for shader_file in [stage[1] for stage in shader_stages]:
        shader_stage = os.path.splitext(shader_file)[1][1:]
        spv_stage_filename = spv_output_dir_name + shader_name + "." + shader_stage + '.spv'
        spv_stages.append(spv_stage_filename)
        spv_generation_command = validation_command[:]
        spv_generation_command.extend(['-V',
                                       '-o', spv_stage_filename,
                                       shader_file])
        (failed, break_validation) = run_command(spv_generation_command,
                                                 shader.name,
                                                 '',
                                                 'Generating SPIR-V binary',
                                                 options.break_on_error)
        failed_validation += failed
        if break_validation:
            break
    return (failed, break_validation, spv_stages)


def link_spirv_stages(shader_name, shader, spv_stages, options):
    """ Link SPIR-V modules into one """
    spv_output_name = './spv/' + shader_name + '.spv'
    linker_command = ['spirv-link']
    linker_command.extend(spv_stages)
    linker_command.extend(['-o', spv_output_name])
    (failed, break_validation) = run_command(linker_command,
                                             shader.name,
                                             '',
                                             'Linking SPIR-V module',
                                             options.break_on_error)
    return (failed, break_validation)


def generate_spirv_file(validation_command, shader, shader_stages, options):
    """ Generate a single SPIR-V module from supplied shader stages """
    shader_name = os.path.splitext(shader.name)[0]  # Remove file extension

    # Create single SPIR-V module per file
    (failed_validation, break_validation, spv_stages) = generate_spirv_stages(
            validation_command,
            shader_name,
            shader,
            shader_stages,
            options)

    if break_validation:
        return (failed_validation, break_validation)

    if not options.unify_into_one_file:
        return (False, break_validation)

    # Link SPIR-V modules into one
    (failed_validation, break_validation) = link_spirv_stages(shader_name,
                                                              shader,
                                                              spv_stages,
                                                              options)

    if not break_validation:
        cleanup_temporary_files(spv_stages)
    return (failed_validation, break_validation)


def validate_shaders(options, shader, shader_stages, failed_validation):
    """ Validates the shader and prints error code if failed """
    break_validation = False

    # Command to call later
    validation_command = ['glslangValidator']

    # Validate compilation of each separate shader
    (number_of_failed_stages, break_validation) = validate_compilation(
                                                    validation_command,
                                                    shader,
                                                    shader_stages,
                                                    options)
    failed_validation += number_of_failed_stages

    if break_validation:
        return (failed_validation, break_validation)

    # Validate linking of all shader stages
    (failed, break_validation) = validate_linking(validation_command,
                                                  shader,
                                                  shader_stages,
                                                  options)
    failed_validation += failed

    if break_validation:
        return (failed_validation, break_validation)

    # Generate a SPIR-V module from all shader stages
    (failed, break_validation) = generate_spirv_file(validation_command,
                                                     shader,
                                                     shader_stages,
                                                     options)
    failed_validation += failed
    return (failed_validation, break_validation)


def cleanup_temporary_files(files):
    """ Remove temporary shader stage files """
    logging.debug('Deleting %s', ', '.join(files))
    try:
        # Clean-up temporary files
        for filename in files:
            os.remove(filename)
    except OSError as exception:
        logging.debug('Unable to remove temporary files: %s %s',
                      exception.strerror,
                      exception.filename)


def write_temporary_shader_stage_files(shader_stages):
    """ Write shader code to temporary files """
    # Write to temporary file and validate it
    for (shader_stage_code, filename) in shader_stages:
        with open(filename, 'w') as file_handle:
            file_handle.write(shader_stage_code)


def do_main_program():
    """ Main program """
    options = parse_cmd_arguments()
    setup_logging(options)

    logging.info("\nValidating GLSL Shaders using Khronos glslangValidator")
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

        shader_stages = identify_shader_stages(shader)
        write_temporary_shader_stage_files(shader_stages)

        # Validate shaders and print output from validator if failed
        (failed_validation, break_validation) = validate_shaders(
            options,
            shader,
            shader_stages,
            failed_validation)
        if break_validation:
            break

        cleanup_temporary_files([stage[1] for stage in shader_stages])

    logging.info("\n%s files in directory validated", str(validated))
    logging.info("%s files failed validation", str(failed_validation))
    if failed_validation:
        sys.exit(-1)


# Main function
try:
    if __name__ == "__main__":
        do_main_program()
except KeyboardInterrupt:
    logging.critical("Keyboard Interrupted!")
