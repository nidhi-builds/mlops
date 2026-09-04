"""
exception.py

Custom exception wrapper so every stage raises errors with the same
format: which file, which line, and what went wrong. Makes it much
easier to debug a failed DVC stage from the terminal output alone.
"""

import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown"
    line_no = exc_tb.tb_lineno if exc_tb else "unknown"
    return f"Error in [{file_name}] at line [{line_no}]: {str(error)}"


class PipelineException(Exception):
    def __init__(self, error: Exception, error_detail: sys = sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message
