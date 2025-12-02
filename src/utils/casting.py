import pathway as pw
import json
from typing import Dict, Any, Optional, Type

# =============================================================================
# Casting UDFs that work on pw.Json fields (like pw.cast)
# =============================================================================
# never add max_batch_size unless you know how to use because it may break the code
@pw.udf(deterministic=True, return_type=str)
def cast_to_str(value: pw.Json) -> str:
    """Cast JSON value to string"""
    if value is None:
        return ""
    return str(value)

@pw.udf(deterministic=True, return_type=int)
def cast_to_int(value: pw.Json) -> int:
    """Cast JSON value to int"""
    if value is None:
        return 0
    return int(value)

@pw.udf(deterministic=True, return_type=float)
def cast_to_float(value: pw.Json) -> float:
    """Cast JSON value to float"""
    if value is None:
        return 0.0
    return float(value)

# =============================================================================
# Main parsing UDF that returns JSON
# =============================================================================

def create_schema_parser(
    schema_class: Type[pw.Schema],
    field_mapping: Optional[Dict[str, str]] = None ,
    default_values: Optional[Dict[str, Any]] = None
):
    field_mapping = field_mapping or {}
    default_values = default_values or {}
    type_casters = {str: str, int: int, float: float, bool: bool}
    
    @pw.udf(deterministic=True, return_type=pw.Json)
    def parser(data: bytes) -> pw.Json:
        try:
            raw_str = data.decode("utf-8")
            parsed = json.loads(raw_str)
            
            result = {}
            for field_name, field_type in schema_class.__annotations__.items():
                source_field = field_mapping.get(field_name, field_name)
                
                if source_field in parsed:
                    value = parsed[source_field]
                elif field_name in default_values:
                    value = default_values[field_name]
                else:
                    raise KeyError(f"Missing: {source_field}")
                
                result[field_name] = type_casters.get(field_type, lambda x: x)(value)
            
            return {"success": 1, "data": result, "error": "", "raw": raw_str}
            
        except Exception as e:
            raw_str = data.decode("utf-8", errors="replace")
            return {
                "success": 0,
                "data": {},
                "error": str(e),
                "raw": raw_str
            }
    
    return parser