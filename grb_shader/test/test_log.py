import grb_shader

logger = grb_shader.setup_log(__name__)

def test_log():
    logger.error('That is a test error message')
    logger.warning('That is a test warning')
    logger.info('That is a test info message')
    logger.debug('That is a test debugging message')