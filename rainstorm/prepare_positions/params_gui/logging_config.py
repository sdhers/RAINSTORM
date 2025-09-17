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
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers to avoid duplicates
        if root_logger.hasHandlers():
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
            try:
                # Main application log
                main_log_file = self.log_dir / f"params_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(detailed_formatter)
                root_logger.addHandler(file_handler)
                
                # Error-only log
                error_log_file = self.log_dir / f"params_gui_errors.log"
                error_handler = logging.handlers.RotatingFileHandler(
                    error_log_file, maxBytes=1024*1024, backupCount=5, encoding='utf-8'
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(detailed_formatter)
                root_logger.addHandler(error_handler)
            except Exception as e:
                logging.error(f"Failed to set up file logging: {e}")

        logging.info("="*50)
        logging.info("Logging system initialized successfully")
        logging.info(f"Log level set to: {logging.getLevelName(self.log_level)}")
        if enable_file_logging:
            logging.info(f"Log directory: {self.log_dir.absolute()}")
        logging.info("="*50)

    def set_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode for detailed logging.
        
        Args:
            enabled: Whether to enable debug mode
        """
        level = logging.DEBUG if enabled else logging.INFO
        logging.getLogger().setLevel(level)
        logging.info(f"Debug mode {'enabled' if enabled else 'disabled'}. Log level set to {logging.getLevelName(level)}.")
    
    def log_system_info(self):
        """Log system information for debugging purposes."""
        import platform
        import customtkinter as ctk
        
        logging.info("=== System Information ===")
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"CustomTkinter version: {ctk.__version__}")
        logging.info("==========================")


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
