# -*- coding: utf-8 -*-
"""
 Copyright (c) 2021 Xin Han
 
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
# coding: utf-8

import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename,
                 level='info',
                 when='D',
                 backCount=25,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(format_str)
        file_handler = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        file_handler.setFormatter(format_str)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)


if __name__ == '__main__':
    log = Logger('all.log', level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('warning')
    log.logger.error('error')
    Logger('error.log', level='error').logger.error('error')
