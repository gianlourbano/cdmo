"""Custom exceptions for STS solver"""


class STSException(Exception):
    """Base exception for STS solver"""
    pass


class InvalidInstanceError(STSException):
    """Raised when instance parameters are invalid"""
    pass


class SolverNotFoundError(STSException):
    """Raised when requested solver is not available"""
    pass


class SolverTimeoutError(STSException):
    """Raised when solver exceeds timeout"""
    pass


class InvalidSolutionError(STSException):
    """Raised when solution validation fails"""
    pass


class ConfigurationError(STSException):
    """Raised when configuration is invalid"""
    pass


class BuildError(STSException):
    """Raised when model building fails"""
    pass
