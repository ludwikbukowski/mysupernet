from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import six

import numpy as np
import time
import json
import warnings
import io
import sys

from collections import deque
from collections import OrderedDict
from collections import Iterable
from keras import backend as K
from keras.callbacks import Callback

class CustomCSVLogger(Callback):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, handicap=0, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.handicap=handicap
        self.writer = None
        self.keys = None
        self.append_header = True
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}
        super(CustomCSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch+self.handicap})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
