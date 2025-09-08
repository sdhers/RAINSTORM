"""
Error handling and user notification utilities for the Rainstorm GUI.

This module provides comprehensive error handling, user-friendly notifications,
and debugging utilities for the parameters editor.
"""

import tkinter as tk
from tkinter import messagebox, ttk
import logging
from typing import Optional, Callable, Any
import threading
import time
from .type_conversion import get_user_friendly_error_message, get_conversion_suggestions

logger = logging.getLogger(__name__)


class ErrorNotificationManager:
    """
    Manages error notifications and user feedback in the GUI.
    Ensures the GUI remains responsive during error conditions.
    """
    
    def __init__(self, parent_window: tk.Tk):
        self.parent = parent_window
        self.notification_queue = []
        self.is_showing_notification = False
        
    def show_conversion_error(self, parameter_name: str, value: str, target_type: str, 
                            suggestions: bool = True, blocking: bool = False):
        """
        Show a user-friendly error message for type conversion failures.
        
        Args:
            parameter_name: Name of the parameter that failed
            value: The invalid value
            target_type: The expected type
            suggestions: Whether to include format suggestions
            blocking: Whether to show as blocking dialog
        """
        title = "Invalid Input"
        message = get_user_friendly_error_message(value, target_type, parameter_name)
        
        if suggestions:
            suggestion_text = get_conversion_suggestions(target_type)
            message += f"\n\n{suggestion_text}"
        
        if blocking:
            messagebox.showerror(title, message, parent=self.parent)
        else:
            self._queue_notification(title, message, "error")
    
    def show_validation_warning(self, parameter_name: str, message: str):
        """
        Show a validation warning to the user.
        
        Args:
            parameter_name: Name of the parameter
            message: Warning message
        """
        title = f"Validation Warning - {parameter_name}"
        self._queue_notification(title, message, "warning")
    
    def show_success_message(self, message: str):
        """
        Show a success message to the user.
        
        Args:
            message: Success message
        """
        self._queue_notification("Success", message, "info")
    
    def _queue_notification(self, title: str, message: str, msg_type: str):
        """
        Queue a notification to be shown without blocking the GUI.
        
        Args:
            title: Notification title
            message: Notification message
            msg_type: Type of message ('error', 'warning', 'info')
        """
        self.notification_queue.append((title, message, msg_type))
        
        if not self.is_showing_notification:
            self._process_notification_queue()
    
    def _process_notification_queue(self):
        """Process queued notifications in a non-blocking way."""
        if not self.notification_queue:
            return
        
        self.is_showing_notification = True
        title, message, msg_type = self.notification_queue.pop(0)
        
        # Schedule the notification to be shown after a brief delay
        # This ensures the GUI remains responsive
        self.parent.after(100, lambda: self._show_notification(title, message, msg_type))
    
    def _show_notification(self, title: str, message: str, msg_type: str):
        """
        Show a notification dialog.
        
        Args:
            title: Notification title
            message: Notification message
            msg_type: Type of message ('error', 'warning', 'info')
        """
        try:
            if msg_type == "error":
                messagebox.showerror(title, message, parent=self.parent)
            elif msg_type == "warning":
                messagebox.showwarning(title, message, parent=self.parent)
            else:
                messagebox.showinfo(title, message, parent=self.parent)
        except Exception as e:
            logger.error(f"Failed to show notification: {e}")
        finally:
            self.is_showing_notification = False
            # Process any remaining notifications
            if self.notification_queue:
                self.parent.after(500, self._process_notification_queue)


class SafeOperationWrapper:
    """
    Wrapper for performing operations safely with proper error handling.
    """
    
    def __init__(self, error_manager: ErrorNotificationManager):
        self.error_manager = error_manager
    
    def safe_execute(self, operation: Callable, operation_name: str, 
                    on_success: Optional[Callable] = None, 
                    on_error: Optional[Callable] = None,
                    show_errors: bool = True) -> Any:
        """
        Execute an operation safely with comprehensive error handling.
        
        Args:
            operation: The operation to execute
            operation_name: Name of the operation for logging
            on_success: Optional callback for successful execution
            on_error: Optional callback for error handling
            show_errors: Whether to show error messages to user
            
        Returns:
            Result of the operation or None if it failed
        """
        try:
            logger.debug(f"Executing safe operation: {operation_name}")
            result = operation()
            
            if on_success:
                on_success(result)
            
            logger.debug(f"Safe operation completed successfully: {operation_name}")
            return result
            
        except Exception as e:
            logger.error(f"Safe operation failed - {operation_name}: {e}")
            
            if show_errors:
                error_msg = f"Operation '{operation_name}' failed: {str(e)}"
                self.error_manager._queue_notification("Operation Failed", error_msg, "error")
            
            if on_error:
                on_error(e)
            
            return None


class ValidationHelper:
    """
    Helper class for input validation with user-friendly feedback.
    """
    
    def __init__(self, error_manager: ErrorNotificationManager):
        self.error_manager = error_manager
    
    def validate_and_convert(self, value: str, target_type: str, parameter_name: str,
                           show_errors: bool = True, fallback_value: Any = None) -> tuple:
        """
        Validate and convert a value with comprehensive error handling.
        
        Args:
            value: Value to validate and convert
            target_type: Target type for conversion
            parameter_name: Name of the parameter for error messages
            show_errors: Whether to show error messages to user
            fallback_value: Fallback value if conversion fails
            
        Returns:
            Tuple of (success: bool, converted_value: Any)
        """
        from .type_conversion import convert_to_type, validate_numeric_input
        
        try:
            # First validate the input
            if not validate_numeric_input(value, target_type):
                if show_errors:
                    self.error_manager.show_conversion_error(
                        parameter_name, value, target_type, suggestions=True
                    )
                return False, fallback_value
            
            # Then convert
            converted = convert_to_type(value, target_type)
            logger.debug(f"Validation and conversion successful: {parameter_name} = {converted}")
            return True, converted
            
        except Exception as e:
            logger.error(f"Validation failed for {parameter_name}: {e}")
            if show_errors:
                self.error_manager.show_conversion_error(
                    parameter_name, value, target_type, suggestions=True
                )
            return False, fallback_value


class DebugInfoCollector:
    """
    Collects debugging information for troubleshooting type conversion issues.
    """
    
    @staticmethod
    def collect_conversion_debug_info() -> dict:
        """
        Collect comprehensive debugging information about type conversions.
        
        Returns:
            Dictionary with debugging information
        """
        from .type_conversion import get_conversion_error_history
        
        debug_info = {
            'timestamp': time.time(),
            'conversion_errors': get_conversion_error_history(),
            'logging_level': logging.getLogger().getEffectiveLevel(),
            'error_count': len(get_conversion_error_history())
        }
        
        return debug_info
    
    @staticmethod
    def export_debug_info(filepath: str):
        """
        Export debugging information to a file.
        
        Args:
            filepath: Path to save the debug information
        """
        import json
        
        try:
            debug_info = DebugInfoCollector.collect_conversion_debug_info()
            
            with open(filepath, 'w') as f:
                json.dump(debug_info, f, indent=2, default=str)
            
            logger.info(f"Debug information exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export debug information: {e}")


class ResponsiveErrorHandler:
    """
    Ensures the GUI remains responsive during error conditions.
    """
    
    def __init__(self, parent_window: tk.Tk):
        self.parent = parent_window
        self.error_count = 0
        self.max_errors_per_session = 50
    
    def handle_error_with_throttling(self, error: Exception, context: str):
        """
        Handle errors with throttling to prevent GUI freezing.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
        """
        self.error_count += 1
        
        # Log the error
        logger.error(f"Error in {context}: {error}")
        
        # Throttle error notifications if too many occur
        if self.error_count > self.max_errors_per_session:
            logger.warning(f"Error throttling activated - too many errors ({self.error_count})")
            return
        
        # Show error in a non-blocking way
        self.parent.after(10, lambda: self._show_throttled_error(str(error), context))
    
    def _show_throttled_error(self, error_msg: str, context: str):
        """Show error message in a throttled, non-blocking way."""
        try:
            if self.error_count <= 10:  # Show first 10 errors normally
                messagebox.showerror("Error", f"Error in {context}:\n{error_msg}", parent=self.parent)
            elif self.error_count <= 20:  # Show next 10 as warnings
                messagebox.showwarning("Multiple Errors", 
                                     f"Multiple errors detected. Latest in {context}:\n{error_msg}", 
                                     parent=self.parent)
            # After 20 errors, only log them
        except Exception as e:
            logger.error(f"Failed to show error dialog: {e}")
    
    def reset_error_count(self):
        """Reset the error count (e.g., when starting a new session)."""
        self.error_count = 0
        logger.info("Error count reset")