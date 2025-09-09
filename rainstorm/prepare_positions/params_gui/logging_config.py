"""
Logging configuration for the Rainstorm Parameters GUI.

This module provides comprehensive logging setup with different levels for
development, debugging, and production use.
"""

import logging
import logging.handlers
from pathlib import Path
import sys
from datetime import datetime


class GUILoggerSetup:
    """
    Centralized logging setup for the Parameters GUI with enhanced error tracking.
    """
    
    def __init__(self, log_dir: Path = None, log_level: str = "INFO"):
        self.log_dir = log_dir or Path("logs")
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.loggers_configured = set()
        
    def setup_logging(self, enable_file_logging: bool = True, enable_console_logging: bool = True):
        """
        Setup comprehensive logging for the GUI application.
        
        Args:
            enable_file_logging: Whether to log to files
            enable_console_logging: Whether to log to console
        """
        # Create logs directory if it doesn't exist
        if enable_file_logging:
            self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        
        # File handlers
        if enable_file_logging:
            # Main application log
            main_log_file = self.log_dir / f"params_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
            
            # Error-only log
            error_log_file = self.log_dir / f"params_gui_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_handler)
            
            # Type conversion specific log
            conversion_log_file = self.log_dir / f"type_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            conversion_handler = logging.FileHandler(conversion_log_file, encoding='utf-8')
            conversion_handler.setLevel(logging.DEBUG)
            conversion_handler.setFormatter(detailed_formatter)
            
            # Add filter to only log type conversion messages
            conversion_handler.addFilter(self._type_conversion_filter)
            root_logger.addHandler(conversion_handler)
        
        # Configure specific loggers
        self._configure_module_loggers()
        
        logging.info("Logging system initialized successfully")
        logging.info(f"Log level: {logging.getLevelName(self.log_level)}")
        if enable_file_logging:
            logging.info(f"Log directory: {self.log_dir.absolute()}")
    
    def _type_conversion_filter(self, record):
        """Filter to only include type conversion related log messages."""
        conversion_modules = [
            'type_conversion',
            'type_registry',
            'error_handling'
        ]
        
        return any(module in record.name for module in conversion_modules)
    
    def _configure_module_loggers(self):
        """Configure specific loggers for different modules."""
        module_configs = {
            'rainstorm.prepare_positions.params_gui.type_conversion': logging.DEBUG,
            'rainstorm.prepare_positions.params_gui.error_handling': logging.DEBUG,
            'rainstorm.prepare_positions.params_gui.sections': logging.INFO,
            'rainstorm.prepare_positions.params_gui.widgets': logging.INFO,
            'rainstorm.prepare_positions.params_gui.params_model': logging.INFO,
            'rainstorm.prepare_positions.params_gui.main_window': logging.INFO,
        }
        
        for module_name, level in module_configs.items():
            logger = logging.getLogger(module_name)
            logger.setLevel(level)
            self.loggers_configured.add(module_name)
    
    def set_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode for detailed logging.
        
        Args:
            enabled: Whether to enable debug mode
        """
        level = logging.DEBUG if enabled else logging.INFO
        
        # Update root logger level
        logging.getLogger().setLevel(level)
        
        # Update all configured loggers
        for logger_name in self.loggers_configured:
            logging.getLogger(logger_name).setLevel(level)
        
        logging.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
    
    def log_system_info(self):
        """Log system information for debugging purposes."""
        import platform
        import tkinter as tk
        
        logging.info("=== System Information ===")
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Python version: {platform.python_version()}")
        
        try:
            root = tk.Tk()
            tk_version = root.tk.eval('info patchlevel')
            root.destroy()
            logging.info(f"Tkinter version: {tk_version}")
        except Exception as e:
            logging.warning(f"Could not determine Tkinter version: {e}")
        
        logging.info("=== End System Information ===")
    
    def create_session_logger(self, session_name: str) -> logging.Logger:
        """
        Create a logger for a specific session or component.
        
        Args:
            session_name: Name of the session/component
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"params_gui.{session_name}")
        logger.setLevel(self.log_level)
        
        # Add session-specific file handler if file logging is enabled
        if self.log_dir.exists():
            session_log_file = self.log_dir / f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            session_handler = logging.FileHandler(session_log_file, encoding='utf-8')
            session_handler.setLevel(self.log_level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            session_handler.setFormatter(formatter)
            logger.addHandler(session_handler)
        
        return logger


def setup_gui_logging(log_level: str = "INFO", log_dir: Path = None, debug_mode: bool = False):
    """
    Convenience function to setup logging for the GUI application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (defaults to ./logs)
        debug_mode: Whether to enable debug mode
    """
    logger_setup = GUILoggerSetup(log_dir, log_level)
    logger_setup.setup_logging()
    
    if debug_mode:
        logger_setup.set_debug_mode(True)
    
    logger_setup.log_system_info()
    
    return logger_setup


def get_conversion_logger() -> logging.Logger:
    """Get a logger specifically for type conversion operations."""
    return logging.getLogger('params_gui.type_conversion')


def get_error_logger() -> logging.Logger:
    """Get a logger specifically for error handling operations."""
    return logging.getLogger('params_gui.error_handling')


def get_gui_logger(component_name: str) -> logging.Logger:
    """
    Get a logger for a specific GUI component.
    
    Args:
        component_name: Name of the GUI component
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f'params_gui.{component_name}')


class LoggingContextManager:
    """
    Context manager for temporary logging level changes.
    """
    
    def __init__(self, logger_name: str, temp_level: int):
        self.logger_name = logger_name
        self.temp_level = temp_level
        self.original_level = None
    
    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level
        logger.setLevel(self.temp_level)
        return logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.original_level)


def with_debug_logging(logger_name: str):
    """
    Context manager to temporarily enable debug logging for a specific logger.
    
    Args:
        logger_name: Name of the logger to enable debug logging for
    """
    return LoggingContextManager(logger_name, logging.DEBUG)


def with_quiet_logging(logger_name: str):
    """
    Context manager to temporarily set logging to ERROR level only.
    
    Args:
        logger_name: Name of the logger to quiet
    """
    return LoggingContextManager(logger_name, logging.ERROR)