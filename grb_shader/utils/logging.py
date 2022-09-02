import logging

# create console handler and set level that is printed on console
grb_shader_console_log_handler = logging.StreamHandler()
#show up to debug messages
grb_shader_console_log_handler.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    
# add formatter to ch
grb_shader_console_log_handler.setFormatter(formatter)

def update_logging_level(level):
    grb_shader_console_log_handler.setLevel(level)

def setup_log(name='grb_shader'):

    # set up logger with name 
    log = logging.getLogger(name)

    # allow debug messages
    log.setLevel(logging.DEBUG)

    # donot duplicate messages
    log.propagate = False  

    # add ch to logger
    log.addHandler(grb_shader_console_log_handler)

    return log
