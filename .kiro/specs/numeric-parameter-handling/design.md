# Design Document

## Overview

The solution involves implementing type-aware parameter handling in the Rainstorm parameters editor GUI. The core issue is that the current implementation uses StringVar objects that always return strings, which are then directly stored in the YAML file without type conversion. The fix will introduce a type inference system that determines the original parameter types and converts string inputs back to their appropriate numeric types before saving.

## Architecture

The solution will be implemented using a layered approach:

1. **Type Registry Layer**: A system to track the original types of parameters as defined in params_builder.py
2. **Type Conversion Layer**: Smart conversion functions that can handle various numeric formats including scientific notation
3. **GUI Integration Layer**: Modified entry creation and saving logic to use type-aware conversion

## Components and Interfaces

### 1. Type Registry System

**Location**: `rainstorm/prepare_positions/params_gui/type_registry.py` (new file)

**Purpose**: Track and provide access to the original parameter types as defined in the params_builder.py

**Key Functions**:
- `get_parameter_type(key_path: List[str]) -> str`: Returns the expected type for a parameter path
- `register_numeric_parameters()`: Builds the registry of numeric parameter paths and their types
- `is_numeric_parameter(key_path: List[str]) -> bool`: Checks if a parameter should be numeric

**Type Registry Structure**:
```python
NUMERIC_PARAMETERS = {
    ('fps',): 'int',
    ('prepare_positions', 'confidence'): 'int', 
    ('prepare_positions', 'median_filter'): 'int',
    ('prepare_positions', 'near_dist'): 'float',
    ('prepare_positions', 'far_dist'): 'float',
    ('prepare_positions', 'max_outlier_connections'): 'int',
    ('geometric_analysis', 'freezing_threshold'): 'float',
    ('geometric_analysis', 'target_exploration', 'distance'): 'int',
    ('geometric_analysis', 'target_exploration', 'orientation', 'degree'): 'int',
    ('automatic_analysis', 'split', 'focus_distance'): 'int',
    ('automatic_analysis', 'split', 'validation'): 'float',
    ('automatic_analysis', 'split', 'test'): 'float',
    ('automatic_analysis', 'RNN', 'units'): 'list_int',
    ('automatic_analysis', 'RNN', 'batch_size'): 'int',
    ('automatic_analysis', 'RNN', 'dropout'): 'float',
    ('automatic_analysis', 'RNN', 'total_epochs'): 'int',
    ('automatic_analysis', 'RNN', 'warmup_epochs'): 'int',
    ('automatic_analysis', 'RNN', 'initial_lr'): 'float',
    ('automatic_analysis', 'RNN', 'peak_lr'): 'float',
    ('automatic_analysis', 'RNN', 'patience'): 'int',
    ('automatic_analysis', 'RNN', 'RNN_width', 'past'): 'int',
    ('automatic_analysis', 'RNN', 'RNN_width', 'future'): 'int',
    ('automatic_analysis', 'RNN', 'RNN_width', 'broad'): 'float',
}
```

### 2. Type Conversion System

**Location**: `rainstorm/prepare_positions/params_gui/type_conversion.py` (new file)

**Purpose**: Provide robust type conversion functions that handle various input formats

**Key Functions**:
- `convert_to_type(value: str, target_type: str) -> Any`: Main conversion function
- `safe_float_conversion(value: str) -> float`: Handles scientific notation and edge cases
- `safe_int_conversion(value: str) -> int`: Handles integer conversion with validation
- `parse_list_values(value: str, element_type: str) -> List`: Handles list parameter conversion

**Conversion Logic**:
- Handle scientific notation (1e-05, 1E-5, etc.)
- Graceful error handling with fallback to original string value
- Support for list parameters with typed elements
- Validation of numeric ranges where appropriate

### 3. Enhanced GUI Integration

**Location**: Modified `rainstorm/prepare_positions/params_gui/sections.py`

**Changes**:
- Modify `_create_entry` method to accept a `parameter_path` argument
- Add type-aware value setting and getting
- Integrate with type registry for automatic type detection

**New Method Signature**:
```python
def _create_entry(self, parent, label_text, data_map, key, comment=None, row=0, field_type='default', parameter_path=None):
```

**Location**: Modified `rainstorm/prepare_positions/params_gui/params_model.py`

**Changes**:
- Enhance the `save()` method to apply type conversion before writing YAML
- Add type conversion pass that processes all parameters before saving

### 4. Widget Enhancement

**Location**: Modified `rainstorm/prepare_positions/params_gui/widgets.py`

**Changes**:
- Update specialized widgets (ROIDataFrame, TargetExplorationFrame, RNNWidthFrame) to use type-aware conversion
- Ensure numeric inputs in these widgets are properly converted

## Data Models

### Parameter Path Structure
Parameters are identified by their path in the YAML hierarchy:
- Top-level: `('fps',)`
- Nested: `('prepare_positions', 'confidence')`
- Deep nested: `('automatic_analysis', 'RNN', 'initial_lr')`

### Type Mapping
- `'int'`: Integer values
- `'float'`: Floating-point values (including scientific notation)
- `'list_int'`: Lists of integers
- `'list_float'`: Lists of floats
- `'str'`: String values (default, no conversion)

## Error Handling

### Conversion Failures
- Log conversion errors for debugging
- Preserve original string value when conversion fails
- Display user-friendly error messages for invalid inputs
- Maintain GUI responsiveness during error conditions

### Type Registry Misses
- Default to string type when parameter not found in registry
- Log missing parameters for future registry updates
- Graceful degradation to current behavior

### Scientific Notation Edge Cases
- Handle various scientific notation formats (e, E, +, -)
- Support very small and very large numbers
- Validate scientific notation syntax

## Testing Strategy

### Unit Tests
**Location**: `tests/test_type_conversion.py` (new file)

**Test Cases**:
- Scientific notation conversion (1e-05, 1E-5, 1.5e-4)
- Integer conversion with edge cases
- Float conversion with various formats
- List parameter conversion
- Error handling for invalid inputs
- Type registry lookup functionality

### Integration Tests
**Location**: `tests/test_gui_type_handling.py` (new file)

**Test Cases**:
- End-to-end parameter editing and saving
- YAML file type preservation after GUI editing
- Backward compatibility with existing params.yaml files
- GUI widget type conversion integration

### Manual Testing Scenarios
1. Edit learning rate parameters (scientific notation)
2. Edit integer parameters (epochs, batch size)
3. Edit float parameters (thresholds, distances)
4. Save and reload parameters to verify type preservation
5. Test with existing params.yaml files to ensure compatibility

## Implementation Approach

### Phase 1: Core Type System
1. Create type_registry.py with parameter type definitions
2. Create type_conversion.py with conversion functions
3. Add comprehensive unit tests for type conversion

### Phase 2: GUI Integration
1. Modify sections.py to use type-aware entry creation
2. Update params_model.py to apply type conversion on save
3. Update specialized widgets for type awareness

### Phase 3: Testing and Validation
1. Add integration tests
2. Manual testing with various parameter types
3. Backward compatibility testing with existing files

### Phase 4: Documentation and Cleanup
1. Update code comments and docstrings
2. Add logging for debugging type conversion issues
3. Performance optimization if needed