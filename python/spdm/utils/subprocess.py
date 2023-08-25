from .logger import logger
import subprocess

def subprocess(cmd, working_dir='.', **kwargs):
    # logger.debug(f"CMD: {cmd} : {res}")
    logger.info(f"Execute Shell command [{working_dir}$ {' '.join(cmd)}]")
    # @ref: https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # env=self.envs,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            check=False,
            **kwargs
        )
        # process_output, _ = command_line_process.communicate()

        with process.stdout as pipe:
            for line in iter(pipe.readline, b''):  # b'\n'-separated lines
                logger.info(line)

        exitcode = process.wait()

    except (OSError, subprocess.CalledProcessError) as error:
        logger.error(
            f"""Command failed! [{cmd}]
                   STDOUT:[{error.stdout}]
                   STDERR:[{error.stderr}]""")
        raise error
    
    return exitcode
