import pathway as pw
from typing import Dict, Any, Optional, Type, List
import json

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

@pw.udf
def json_to_bytes(obj: pw.Json) -> bytes:
    return json.dumps(obj).encode("utf-8")

# =============================================================================
# Main parsing UDF that returns JSON
# =============================================================================

import pathway as pw
from typing import Type, Optional, Dict, Any


def create_schema_parser(
    schema_class: Type[pw.Schema],
    field_mapping: Optional[Dict[str, str]] = None,
    default_values: Optional[Dict[str, Any]] = None
):
    field_mapping = field_mapping or {}
    default_values = default_values or {}
    type_casters = {str: str, int: int, float: float, bool: bool}
    schema_fields = list(schema_class.__annotations__.items())
    
    @pw.udf
    def parser(data: pw.Json) -> pw.Json:
        try:
            if data is {}:
                raise ValueError("data is None")
            result = {}
            for field_name, field_type in schema_fields:
                source_field = field_mapping.get(field_name, field_name)
                
                # Direct access with try-except (avoids 'in' operator issues)
                try:
                    value = data[source_field]
                except KeyError:
                    if field_name in default_values:
                        value = default_values[field_name]
                    else:
                        raise KeyError(f"Missing: {source_field}")
                
                caster = type_casters.get(field_type, lambda x: x)
                result[field_name] = caster(value)
            
            # Don't include raw pw.Json in return - causes serialization error
            return {"success": 1, "data": result, "error": ""}
            
        except Exception as e:
            return {"success": 0, "data": {}, "error": str(e)}
    
    return parser


def unpack_from_schema(
    table: pw.Table,
    schema_class: Type[pw.Schema],
    source_column: str = "data",
    field_mapping: Optional[Dict[str, str]] = None,
) -> pw.Table:
    field_mapping = field_mapping or {}
    type_casters = {str: str, int: int, float: float, bool: bool}
    
    source = getattr(pw.this, source_column)
    select_kwargs = {}
    
    for field_name, field_type in schema_class.__annotations__.items():
        source_field = field_mapping.get(field_name, field_name)
        caster = type_casters.get(field_type, lambda x: x)
        select_kwargs[field_name] = pw.apply(caster, source[source_field])
    
    return table.select(**select_kwargs)



def dedupe_from_schema(
    table: pw.Table,
    schema_class: Type[pw.Schema],
    dedupe_columns: List[str],
) -> pw.Table:
    fields = list(schema_class.__annotations__.keys())
    
    groupby_refs = [getattr(pw.this, col) for col in dedupe_columns]
    
    reduce_kwargs = {
        field: pw.reducers.earliest(getattr(pw.this, field))
        for field in fields
    }
    
    return table.groupby(*groupby_refs).reduce(**reduce_kwargs)