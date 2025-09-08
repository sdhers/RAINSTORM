# Requirements Document

## Introduction

The Rainstorm parameters editor currently has an issue where numeric parameters (integers and floats) are being stored as strings in the params.yaml file when edited through the GUI. This causes problems in the notebook 3a-Create_models.ipynb which expects these values to be proper numeric types. The issue occurs because the GUI uses StringVar objects that always return strings, and these string values are directly stored in the YAML without type conversion.

## Requirements

### Requirement 1

**User Story:** As a researcher using Rainstorm, I want numeric parameters to maintain their proper data types (int/float) when I edit them through the parameters editor, so that the downstream notebooks work correctly without type errors.

#### Acceptance Criteria

1. WHEN a user edits a numeric parameter in the GUI THEN the system SHALL store the value as the appropriate numeric type (int or float) in the params.yaml file
2. WHEN the params.yaml file is loaded by downstream notebooks THEN numeric parameters SHALL be available as proper numeric types, not strings
3. WHEN a parameter is defined as a float in the original params_builder.py THEN it SHALL remain a float after GUI editing
4. WHEN a parameter is defined as an integer in the original params_builder.py THEN it SHALL remain an integer after GUI editing

### Requirement 2

**User Story:** As a developer maintaining the Rainstorm codebase, I want the type conversion to happen automatically without requiring changes to existing parameter definitions, so that the fix is backward compatible and doesn't break existing functionality.

#### Acceptance Criteria

1. WHEN the fix is implemented THEN existing parameter definitions in params_builder.py SHALL continue to work without modification
2. WHEN the GUI saves parameters THEN the type conversion SHALL happen automatically based on the original parameter types
3. WHEN a user enters invalid numeric input THEN the system SHALL handle the error gracefully and preserve the previous valid value
4. WHEN the system cannot determine the original type THEN it SHALL default to preserving the string value to avoid data loss

### Requirement 3

**User Story:** As a researcher, I want scientific notation values (like 1e-05) to be properly handled and preserved as floats, so that learning rate parameters and other small values work correctly.

#### Acceptance Criteria

1. WHEN a parameter contains scientific notation (e.g., 1e-05) THEN it SHALL be stored as a float value
2. WHEN a user edits a scientific notation value through the GUI THEN it SHALL maintain its numeric type
3. WHEN the YAML file displays scientific notation THEN the GUI SHALL correctly parse and display it as an editable numeric value
4. WHEN a user enters scientific notation in the GUI THEN it SHALL be properly converted to a float