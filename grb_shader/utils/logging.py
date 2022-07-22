import logging

def setup_log(name):

    # set up logger with name 
    log = logging.getLogger(name)

    # allow debug messages
    log.setLevel(logging.DEBUG)

    # donot duplicate messages
    log.propagate = False

    return log
