import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    """
    Base class for all exceptions in the Network Security module.
    This class extends the built-in Exception class.
    """
    def __init__(self, error_message,error_details:sys):
        """
        Initializes the NetworkSecurityException with an error message and details.
        
        :param error_message: A string containing the error message.
        :param error_details: An object containing details about the error.
        """

        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
            """
            Returns a string representation of the exception.
            This includes the error message, filename, and line number.
            """

            return "Error occurred in script name [{0}] at line number [{1}] with error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)  )   

# if __name__ == "__main__":
#     try:
#         logger.logging.info("This is a test log message.")
#         a= 1 / 0
#         print("This will not be printed",a)
#     except Exception as e:
#         raise NetworkSecurityException(e, sys)