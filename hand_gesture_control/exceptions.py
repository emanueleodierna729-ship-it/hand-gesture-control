"""Custom exceptions for Hand Gesture Control System."""


class GestureControlException(Exception):
    """Base exception for all gesture control errors."""
    pass


class CameraException(GestureControlException):
    """Raised when camera initialization or capture fails."""
    pass


class GestureRecognitionException(GestureControlException):
    """Raised when gesture recognition fails."""
    pass


class DatabaseException(GestureControlException):
    """Raised when gesture database operations fail."""
    pass


class VoiceControlException(GestureControlException):
    """Raised when voice recognition fails."""
    pass


class MouseControlException(GestureControlException):
    """Raised when mouse/keyboard control fails."""
    pass


class ConfigurationException(GestureControlException):
    """Raised when configuration validation fails."""
    pass
