"""
Reusable tool-argument validation for any system under test.

    from validated_tool import validated_tool, validated_function

    class RefundArgs(BaseModel):
        order_total: float = Field(ge=0)
        days_since_purchase: int = Field(ge=0)

    safe_tool = validated_tool(refund_calculator, RefundArgs)   # LangChain tool
    safe_fn = validated_function(my_fn, RefundArgs)             # plain Python function

The wrapped tool checks its arguments against the Pydantic Field rules
before running, and turns any bad input or exception into a clean error
string the model can read and recover from. It never raises:

    "Error: invalid input for refund_calculator: days_since_purchase: Input should be
     greater than or equal to 0 (got -3). Fix the input and try again."

What the model is TOLD about the tool (name, description, argument schema)
is unchanged: constraints are checked inside the wrapper, not added to the
published schema, so switching validation on changes one thing only.
"""

from pydantic import BaseModel, ConfigDict, ValidationError


class StrictArgs(BaseModel):
    """Optional base for constraint models: strips whitespace before checking."""
    model_config = ConfigDict(str_strip_whitespace=True)


def _format(e: ValidationError, constraints) -> str:
    parts = []
    for err in e.errors():
        field = ".".join(str(p) for p in err["loc"]) or "input"
        msg = err["msg"]
        if err["type"] == "missing":
            parts.append(f"{field}: required but missing")
            continue
        info = constraints.model_fields.get(str(err["loc"][0])) if constraints and err["loc"] else None
        hint = f" (expected: {info.description})" if info is not None and info.description else ""
        got = err.get("input")
        got_text = f" (got {got!r})" if got is not None and not isinstance(got, dict) else ""
        parts.append(f"{field}: {msg}{hint}{got_text}")
    return "; ".join(parts)


def check(constraints, kwargs) -> str | None:
    """None if kwargs satisfy the constraints, else a readable error."""
    try:
        constraints(**kwargs)
        return None
    except ValidationError as e:
        return _format(e, constraints)


def _invalid(name, detail):
    return f"Error: invalid input for {name}: {detail}. Fix the input and try again."


def validated_function(fn, constraints, name=None):
    """Plain-function version: fn(**kwargs) with validation and no exceptions escaping."""
    name = name or getattr(fn, "__name__", "tool")

    def wrapped(**kwargs):
        error = check(constraints, kwargs)
        if error:
            return _invalid(name, error)
        try:
            return fn(**kwargs)          # the ORIGINAL arguments: valid input -> identical output
        except Exception as e:           # a tool bug must not crash the whole run
            return f"Error: {name} failed ({type(e).__name__}: {e})"

    wrapped.__name__ = f"validated_{name}"
    wrapped.__wrapped__ = fn
    return wrapped


def validated_tool(tool, constraints):
    """A copy of a LangChain tool with the same name, description and argument schema,
    whose function is validated_function(...). Type errors / missing arguments that
    LangChain catches before the function runs also come back as a clean string."""
    name = tool.name

    def on_schema_error(e):
        return _invalid(name, _format(e, constraints) if isinstance(e, ValidationError) else str(e))

    return tool.model_copy(update={"func": validated_function(tool.func, constraints, name),
                                   "handle_validation_error": on_schema_error})
